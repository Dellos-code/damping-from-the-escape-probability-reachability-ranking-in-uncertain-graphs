"""
Does the sampling advantage survive scale?

Hypothesis.  A Monte-Carlo estimate of y* has per-node error O(1/sqrt(W)),
independent of n.  But as n grows the y* values crowd together, so a FIXED
sample budget corrupts an increasing fraction of ranking pairs.  A deterministic
diffusion has no such term.  If that is right, MC's lead should shrink with n
while the diffusion's accuracy holds, and there is some n beyond which the
deterministic score wins.

Design.  BFS-sampled induced subgraphs of one real network at increasing n, with
the edge probability tuned per instance so that Pesc_bar hits a FIXED target.
That holds the density regime constant and varies only n, which is the only way
to attribute any trend to scale rather than to regime.

Usage:  python3 scale_real.py <genre> <target_pesc> <n1,n2,...>
Appends to scale_real.jsonl.
"""

import json
import sys
import time
import numpy as np
import scipy.sparse as sp
import networkx as nx

sys.path.insert(0, "..")
from escape_lib import (Instance, ground_truth, rho, timed, diffuse,
                        alpha_from_rule, m_pesc, m_irie, m_mc, m_escape_pull)

CAL = json.load(open("../calibration.json"))
COEF = (CAL["intercept"], CAL["slope"])
AT = json.load(open("../alpha_transfer.json"))
PUSH_COEF = tuple(AT["rules"]["PPR push"])
OUT = "scale_real.jsonl"

FILES = {"coauthor-physics": "ms_academic_phy.npz",
         "coauthor-cs": "ms_academic_cs.npz",
         "amazon-computers": "amazon_electronics_computers.npz",
         "amazon-photo": "amazon_electronics_photo.npz"}


def load_graph(genre):
    d = np.load(FILES[genre], allow_pickle=True)
    A = sp.csr_matrix((d["adj_data"], d["adj_indices"], d["adj_indptr"]),
                      shape=tuple(d["adj_shape"]))
    A = ((A + A.T) > 0).astype(np.int8)
    A.setdiag(0)
    A.eliminate_zeros()
    return nx.from_scipy_sparse_array(A)


def bfs_ball(G, n_target, seed):
    """Snowball sample: preserves local density far better than a random
    node sample, which would artificially sparsify the subgraph."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    start = int(nodes[rng.integers(len(nodes))])
    seen = {start}
    frontier = [start]
    while len(seen) < n_target and frontier:
        nxt = []
        for u in frontier:
            for v in G.neighbors(u):
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
                    if len(seen) >= n_target:
                        break
            if len(seen) >= n_target:
                break
        frontier = nxt
    return nx.convert_node_labels_to_integers(G.subgraph(seen).copy())


def edges_of(G):
    ei, ej = [], []
    for u, v in G.edges():
        a, b = (u, v) if u < v else (v, u)
        ei.append(a)
        ej.append(b)
    return np.array(ei, dtype=np.int32), np.array(ej, dtype=np.int32)


def tune_p(n, ei, ej, target, x):
    """Binary-search a constant edge probability giving mean Pesc == target,
    so every size sits in the same density regime."""
    lo, hi = 1e-5, 0.999
    for _ in range(40):
        mid = (lo + hi) / 2
        inst = Instance(n, ei, ej, np.full(len(ei), mid), x)
        if inst.mean_pesc < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run(genre, target, sizes):
    G = load_graph(genre)
    print(f"{genre}: n={G.number_of_nodes()} m={G.number_of_edges()}")
    with open(OUT, "a") as fh:
        for n_t in sizes:
            t0 = time.time()
            S = bfs_ball(G, n_t, seed=11)
            n = S.number_of_nodes()
            ei, ej = edges_of(S)
            rng = np.random.default_rng(5)
            x = rng.dirichlet(np.full(n, 0.7))
            x /= x.sum()
            p = tune_p(n, ei, ej, target, x)
            inst = Instance(n, ei, ej, np.full(len(ei), p), x)

            gtA = ground_truth(inst, 3000, seed=101)
            gtB = ground_truth(inst, 3000, seed=202)   # GT reliability check
            gt_self = rho(gtA, gtB)

            a_e = alpha_from_rule(inst, COEF)
            a_p = float(np.clip(PUSH_COEF[0] + PUSH_COEF[1] * inst.mean_pesc,
                                0.4, 1.0))
            METH = {
                "P_esc alone":   lambda i: m_pesc(i),
                "IRIE":          lambda i: m_irie(i, 0.7),
                "escape rule-a": lambda i: m_escape_pull(i, a_e),
                "push PPR rule-a": lambda i: diffuse(i, a_p, use_pesc=False,
                                                     orientation="push"),
                "MC 50":         lambda i: m_mc(i, 50),
                "MC 200":        lambda i: m_mc(i, 200),
            }
            rec = dict(genre=genre, target=target, n=n, m=int(inst.m),
                       p=float(p), mean_pesc=float(inst.mean_pesc),
                       gt_self_consistency=float(gt_self), rho={}, ms={})
            for nm, fn in METH.items():
                s, dt = timed(fn, inst)
                rec["rho"][nm] = rho(s, gtA)
                rec["ms"][nm] = dt * 1000
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  n={n:6d} m={inst.m:7d} p={p:.4f} Pesc={inst.mean_pesc:.3f} "
                  f"| GT self-consistency {gt_self:.4f} | "
                  f"MC200 {rec['rho']['MC 200']:.3f} "
                  f"escape {rec['rho']['escape rule-a']:.3f} "
                  f"IRIE {rec['rho']['IRIE']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], float(sys.argv[2]),
        [int(z) for z in sys.argv[3].split(",")])
