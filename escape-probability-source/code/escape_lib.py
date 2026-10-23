"""
escape_lib.py
Core library for "Escape-Scaled Diffusion for Reachability Ranking in
Probabilistic Graphs with Uncertain Targets".

Contains:
  * instance generation (ER / BA / grid / Watts-Strogatz potential supports,
    with independent edge-presence probabilities drawn from a range)
  * Monte-Carlo ground truth for y*_i = E[ sum_{v in C(i)} x_v ]
  * every ranking method under comparison, including the influence-maximisation
    proxies (IRIE, MIA, degree-discount) and the sampling estimators
    (RIS / plain Monte Carlo) that the first version of this note omitted.

All scoring functions share the signature  f(inst, **kw) -> np.ndarray of scores.
"""

import time
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

# --------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------


class Instance:
    """An uncertain graph with an uncertain-target prior.

    Attributes
    ----------
    n     : number of vertices
    ei,ej : endpoint arrays of the m undirected potential edges (ei < ej)
    p     : per-edge presence probability, shape (m,)
    x     : target prior, shape (n,), sums to 1
    P     : (n,n) sparse symmetric matrix of presence probabilities
    """

    def __init__(self, n, ei, ej, p, x, meta=None):
        self.n = n
        self.ei = np.asarray(ei)
        self.ej = np.asarray(ej)
        self.p = np.asarray(p, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.meta = meta or {}
        self.m = len(self.p)
        # symmetric probability matrix
        rows = np.concatenate([self.ei, self.ej])
        cols = np.concatenate([self.ej, self.ei])
        vals = np.concatenate([self.p, self.p])
        self.P = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        self._pesc = None
        self._W_cache = {}

    # ---- cheap statistics -------------------------------------------------
    @property
    def pesc(self):
        """P_esc(i) = 1 - prod_j (1 - p_ij): probability i is not isolated."""
        if self._pesc is None:
            L = sp.csr_matrix(
                (np.log1p(-np.clip(self.P.data, 0, 1 - 1e-15)),
                 self.P.indices, self.P.indptr),
                shape=self.P.shape,
            )
            self._pesc = 1.0 - np.exp(np.asarray(L.sum(axis=1)).ravel())
        return self._pesc

    @property
    def mean_pesc(self):
        return float(self.pesc.mean())

    def W(self, weighting="log"):
        """Row-stochastic transition matrix on the potential support."""
        if weighting in self._W_cache:
            return self._W_cache[weighting]
        d = self.P.data
        if weighting == "log":
            w = -np.log1p(-np.clip(d, 0, 1 - 1e-15))
        elif weighting == "linear":
            w = d.copy()
        elif weighting == "uniform":
            w = np.ones_like(d)
        else:
            raise ValueError(weighting)
        A = sp.csr_matrix((w, self.P.indices, self.P.indptr), shape=self.P.shape)
        rs = np.asarray(A.sum(axis=1)).ravel()
        inv = np.where(rs > 0, 1.0 / np.where(rs > 0, rs, 1.0), 0.0)
        Wm = sp.diags(inv) @ A
        self._W_cache[weighting] = Wm.tocsr()
        return self._W_cache[weighting]


def _prior(n, rng, kind="dirichlet"):
    if kind == "dirichlet":
        x = rng.dirichlet(np.full(n, 0.7))
    elif kind == "uniform":
        x = np.full(n, 1.0 / n)
    elif kind == "sparse":
        x = np.zeros(n)
        k = max(1, n // 10)
        idx = rng.choice(n, size=k, replace=False)
        x[idx] = rng.dirichlet(np.full(k, 1.0))
    else:
        raise ValueError(kind)
    return x / x.sum()


def make_instance(topology, n, p_lo, p_hi, seed, prior_kind="dirichlet",
                  avg_deg=6):
    """Build a potential support of the requested family and attach edge
    probabilities drawn uniformly from [p_lo, p_hi] and a target prior."""
    rng = np.random.default_rng(seed)
    if topology == "er":
        G = nx.gnp_random_graph(n, avg_deg / (n - 1), seed=int(seed))
    elif topology == "ba":
        G = nx.barabasi_albert_graph(n, max(1, avg_deg // 2), seed=int(seed))
    elif topology == "grid":
        side = int(round(np.sqrt(n)))
        assert side * side == n, f"grid needs square n, got {n}"
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    elif topology == "ws":
        G = nx.watts_strogatz_graph(n, avg_deg, 0.1, seed=int(seed))
    else:
        raise ValueError(topology)
    ei, ej = [], []
    for u, v in G.edges():
        a, b = (u, v) if u < v else (v, u)
        ei.append(a)
        ej.append(b)
    ei = np.array(ei, dtype=np.int32)
    ej = np.array(ej, dtype=np.int32)
    p = rng.uniform(p_lo, p_hi, size=len(ei))
    x = _prior(n, rng, prior_kind)
    return Instance(n, ei, ej, p, x,
                    meta=dict(topology=topology, n=n, p_lo=p_lo, p_hi=p_hi,
                              seed=seed, prior=prior_kind))


# --------------------------------------------------------------------------
# Ground truth  (== single-seed weighted influence spread under IC,
#                == weighted Edge Percolated Component score)
# --------------------------------------------------------------------------


def _sample_component_mass(inst, rng):
    """One possible world; return the component mass vector."""
    keep = rng.random(inst.m) < inst.p
    ei, ej = inst.ei[keep], inst.ej[keep]
    A = sp.coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(inst.n, inst.n))
    ncomp, lab = connected_components(A, directed=False)
    mass = np.bincount(lab, weights=inst.x, minlength=ncomp)
    return mass[lab]


def ground_truth(inst, n_samples=3000, seed=0):
    """Monte-Carlo estimate of y*_i = E[ sum_{v in C(i)} x_v ]."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(inst.n)
    for _ in range(n_samples):
        acc += _sample_component_mass(inst, rng)
    return acc / n_samples


# --------------------------------------------------------------------------
# Ranking methods
# --------------------------------------------------------------------------


def _normalise(v):
    s = v.sum()
    return v / s if s > 0 else v


def diffuse(inst, alpha, use_pesc=True, orientation="pull", weighting="log",
            tol=1e-13, max_iter=800):
    """The iteration of the paper.

        pull:  y <- normalise[(1-a) x + a (Pesc . (W y))]
        push:  y <- normalise[(1-a) x + a (Pesc . (W^T y))]

    With use_pesc=False and orientation='push' this is standard personalised
    PageRank; with use_pesc=False and orientation='pull' it is its adjoint.
    """
    W = inst.W(weighting)
    Op = W if orientation == "pull" else W.T.tocsr()
    pe = inst.pesc if use_pesc else 1.0
    x = inst.x
    y = x.copy()
    for _ in range(max_iter):
        z = Op @ y
        if use_pesc:
            z = pe * z
        y_new = _normalise((1.0 - alpha) * x + alpha * z)
        if np.abs(y_new - y).max() < tol:
            y = y_new
            break
        y = y_new
    return y


ALPHA_RULE = (0.544, 0.494)


def alpha_from_rule(inst, coef=ALPHA_RULE):
    a, b = coef
    return float(np.clip(a + b * inst.mean_pesc, 0.4, 1.0))


# ---- trivial / structural baselines ---------------------------------------

def m_prior(inst):
    return inst.x.copy()


def m_sum_p(inst):
    return np.asarray(inst.P.sum(axis=1)).ravel()


def m_pesc(inst):
    return inst.pesc.copy()


def m_degree(inst):
    """Plain potential degree."""
    return np.asarray((inst.P > 0).sum(axis=1)).ravel().astype(float)


def m_degree_discount(inst):
    """Chen, Wang & Yang (KDD 2009), DegreeDiscountIC, evaluated for a
    single seed.  The discount term t_v counts already-selected neighbours;
    for |S| = 1 it is identically zero, so the score reduces exactly to the
    (weighted) degree.  Implemented explicitly so the reduction can be
    verified numerically rather than merely asserted."""
    t = np.zeros(inst.n)                       # no seed selected yet
    d = np.asarray(inst.P.sum(axis=1)).ravel()  # expected degree
    pbar = np.asarray(inst.P.sum(axis=1)).ravel() / np.maximum(
        1, np.asarray((inst.P > 0).sum(axis=1)).ravel())
    return d - 2 * t - (d - t) * t * pbar


# ---- PageRank family ------------------------------------------------------

def m_ppr_pull(inst, alpha=0.85, weighting="log"):
    return diffuse(inst, alpha, use_pesc=False, orientation="pull",
                   weighting=weighting)


def m_ppr_push(inst, alpha=0.85, weighting="log"):
    return diffuse(inst, alpha, use_pesc=False, orientation="push",
                   weighting=weighting)


def m_pull_times_pesc(inst, alpha=None, weighting="log"):
    a = alpha_from_rule(inst) if alpha is None else alpha
    return diffuse(inst, a, use_pesc=False, orientation="pull",
                   weighting=weighting) * inst.pesc


def m_escape_pull(inst, alpha=None, weighting="log"):
    a = alpha_from_rule(inst) if alpha is None else alpha
    return diffuse(inst, a, use_pesc=True, orientation="pull",
                   weighting=weighting)


# ---- influence-maximisation proxies (the missing baselines) ---------------

def _spectral_radius(M, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(M.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = M @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0
        v = w / nw
        lam = nw
    return float(lam)


def m_irie(inst, alpha=0.7, n_iter=400, tol=1e-12, safety=0.95):
    """IRIE influence rank (Jung, Heo & Chen, ICDM 2012), node-weighted.

    Original:      r(u) = 1 + a * sum_v p_uv r(v)
    Weighted here: r    = x + a * P r        (x is the target prior)

    P is the raw probability matrix, whose row sums may exceed one; that
    departure from stochasticity is exactly IRIE's design point.  It also
    means the Neumann series diverges when a * rho(P) >= 1.  IRIE's default
    a = 0.7 assumes weighted-cascade probabilities (p ~ 1/deg, row sums ~ 1);
    our p_ij are drawn from a fixed range, so we clamp
        a_eff = min(a, safety / rho(P))
    which is the natural spectral analogue of the original setting and keeps
    the comparison fair rather than letting the baseline blow up.
    """
    rp = _spectral_radius(inst.P)
    a_eff = min(alpha, safety / rp) if rp > 0 else alpha
    x = inst.x
    r = x.copy()
    for _ in range(n_iter):
        r_new = x + a_eff * (inst.P @ r)
        if np.abs(r_new - r).max() < tol * max(1.0, np.abs(r).max()):
            r = r_new
            break
        r = r_new
    return r


def m_irie_wc(inst, alpha=0.7, n_iter=400, tol=1e-12, weighting="linear"):
    """IRIE on a row-normalised support -- the weighted-cascade regime the
    method was designed for (row sums exactly 1, so any alpha < 1 converges).
    """
    W = inst.W(weighting)
    x = inst.x
    r = x.copy()
    for _ in range(n_iter):
        r_new = x + alpha * (W @ r)
        if np.abs(r_new - r).max() < tol * max(1.0, np.abs(r).max()):
            r = r_new
            break
        r = r_new
    return r


def m_mia(inst, theta=1e-3):
    """MIA / PMIA-style score (Chen, Wang & Wang, KDD 2010), node-weighted.

    sigma(u) ~= sum_v x_v * pp(u,v) where pp(u,v) is the maximum-probability
    path, truncated at pp > theta.  Computed as a Dijkstra on -log p.
    """
    d = np.clip(inst.P.data, 1e-15, 1.0)
    Wd = sp.csr_matrix((-np.log(d), inst.P.indices, inst.P.indptr),
                       shape=inst.P.shape)
    from scipy.sparse.csgraph import dijkstra
    lim = -np.log(theta)
    D = dijkstra(Wd, directed=False, limit=lim)
    PP = np.exp(-D)
    PP[~np.isfinite(D)] = 0.0
    np.fill_diagonal(PP, 1.0)
    return PP @ inst.x


# ---- sampling estimators (the honest topline) -----------------------------

def m_mc(inst, n_samples=50, seed=12345):
    """Plain Monte-Carlo estimate of y* with a small sample budget.
    This is the same estimator as the ground truth, just cheaper."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(inst.n)
    for _ in range(n_samples):
        acc += _sample_component_mass(inst, rng)
    return acc / n_samples


def m_ris(inst, n_samples=50, seed=999):
    """Reverse-reachable-set estimator (Borgs et al., SODA 2014) specialised
    to this objective.  Sample v ~ x, sample a world, take RR(v) = C(v);
    then P(u in RR) = sum_v x_v P(u ~ v) = y*_u exactly."""
    rng = np.random.default_rng(seed)
    hits = np.zeros(inst.n)
    for _ in range(n_samples):
        v = rng.choice(inst.n, p=inst.x)
        keep = rng.random(inst.m) < inst.p
        ei, ej = inst.ei[keep], inst.ej[keep]
        A = sp.coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(inst.n, inst.n))
        _, lab = connected_components(A, directed=False)
        hits += (lab == lab[v])
    return hits / n_samples


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------


def rho(score, truth):
    r = spearmanr(score, truth).statistic
    return 0.0 if not np.isfinite(r) else float(r)


def timed(fn, inst, repeats=1, **kw):
    t0 = time.perf_counter()
    for _ in range(repeats):
        s = fn(inst, **kw)
    dt = (time.perf_counter() - t0) / repeats
    return s, dt


def oracle_alpha(inst, truth, grid=None, weighting="log"):
    grid = grid if grid is not None else np.round(np.arange(0.40, 1.001, 0.05), 3)
    best, ba = -2.0, 1.0
    for a in grid:
        r = rho(m_escape_pull(inst, alpha=float(a), weighting=weighting), truth)
        if r > best:
            best, ba = r, float(a)
    return ba, best
