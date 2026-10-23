"""
Message-passing percolation (Karrer, Newman & Zdeborova, PRL 113:208702, 2014).

For each directed edge i->j let nu_{i->j} be the probability that i connects to
the giant cluster through some neighbour other than j:

    nu_{i->j} = 1 - prod_{k in N(i)\\j} (1 - p_ik * nu_{k->i})

and the probability that vertex i belongs to the giant cluster is

    mu_i      = 1 - prod_{k in N(i)}   (1 - p_ik * nu_{k->i}).

Initialising nu == 1 and evaluating mu once, with no message updates, gives

    mu_i = 1 - prod_k (1 - p_ik) = Pesc(i),

so the escape probability is exactly the correlation-free (eta = 1) limit of
this recursion -- its zeroth iterate.  That identity is why Pesc is treated in
this work as a free proxy for message passing rather than as a mechanism of its
own.

Products are evaluated in log space; the excluded-neighbour term is removed by
subtraction rather than division, which is stable when p*nu approaches 1.
"""

import numpy as np


def _directed(inst):
    """Two directed edges per undirected edge; rev[2t]=2t+1, rev[2t+1]=2t."""
    ei, ej, p = inst.ei, inst.ej, inst.p
    src = np.empty(2 * len(ei), dtype=np.int64)
    dst = np.empty(2 * len(ei), dtype=np.int64)
    pe = np.empty(2 * len(ei), dtype=float)
    src[0::2], dst[0::2] = ei, ej
    src[1::2], dst[1::2] = ej, ei
    pe[0::2] = p
    pe[1::2] = p
    rev = np.arange(2 * len(ei)) ^ 1
    return src, dst, pe, rev


def mp_percolation(inst, n_iter=200, tol=1e-10, damping=0.0, return_iters=False):
    """Per-node probability of belonging to the giant cluster."""
    n = inst.n
    src, dst, pe, rev = _directed(inst)
    if len(pe) == 0:
        out = np.zeros(n)
        return (out, 0) if return_iters else out
    nu = np.ones(len(pe))
    EPS = 1e-15
    it = 0
    for it in range(1, n_iter + 1):
        # contribution of edge (k->i) to node i = log(1 - p * nu_{k->i})
        contrib = np.log(np.clip(1.0 - pe * nu, EPS, 1.0))
        # L[i] = sum over all incoming edges k->i
        L = np.bincount(dst, weights=contrib, minlength=n)
        # for edge e = i->j : exclude the term arriving on rev[e] = j->i
        nu_new = 1.0 - np.exp(L[src] - contrib[rev])
        nu_new = np.clip(nu_new, 0.0, 1.0)
        if damping > 0:
            nu_new = damping * nu + (1 - damping) * nu_new
        delta = np.abs(nu_new - nu).max()
        nu = nu_new
        if delta < tol:
            break
    contrib = np.log(np.clip(1.0 - pe * nu, EPS, 1.0))
    L = np.bincount(dst, weights=contrib, minlength=n)
    mu = 1.0 - np.exp(L)
    return (mu, it) if return_iters else mu


def m_mp(inst, n_iter=200, tol=1e-10):
    """Ranking score: giant-cluster membership probability."""
    return mp_percolation(inst, n_iter=n_iter, tol=tol)


def m_mp_weighted(inst, n_iter=200, tol=1e-10):
    """Membership probability scaled by the prior mass it would reach.

    y*_i ~= mu_i * (prior mass of the giant cluster) + (local mass if not in it).
    The first term is a constant across i, so for RANKING mu_i alone is the
    relevant quantity; this variant adds the second-order local term.
    """
    mu = mp_percolation(inst, n_iter=n_iter, tol=tol)
    giant_mass = float(np.dot(mu, inst.x))
    return mu * giant_mass + (1.0 - mu) * inst.x


def m_mp_k(inst, k):
    """Message passing truncated at k iterations.  k=0 is exactly Pesc."""
    n = inst.n
    src, dst, pe, rev = _directed(inst)
    if len(pe) == 0:
        return np.zeros(n)
    nu = np.ones(len(pe))
    EPS = 1e-15
    for _ in range(k):
        contrib = np.log(np.clip(1.0 - pe * nu, EPS, 1.0))
        L = np.bincount(dst, weights=contrib, minlength=n)
        nu = np.clip(1.0 - np.exp(L[src] - contrib[rev]), 0.0, 1.0)
    contrib = np.log(np.clip(1.0 - pe * nu, EPS, 1.0))
    return 1.0 - np.exp(np.bincount(dst, weights=contrib, minlength=n))
