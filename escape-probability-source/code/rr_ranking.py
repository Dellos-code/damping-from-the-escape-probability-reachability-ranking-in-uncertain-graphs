"""
Reverse-reachable-set ranking estimator (the RIS framework).

Estimator.  Sample a root r ~ x; sample a possible world lazily; the
reverse-reachable set is RR(r) = C(r), the connected component of r.  Then

    E[ 1{ i in RR } ]  =  sum_r x_r P(i ~ r)  =  y*_i        (exactly),

so the fraction of theta sets containing i is an unbiased estimate of the
expected reachable prior mass.  This is the singleton-spread special case of
reverse-reachable-set sampling (Borgs et al., SODA 2014; Tang et al., SIGMOD
2015), with the root drawn from the node-weight prior as in weighted extensions.

Construction.  Sets are generated lazily and level-synchronously: only edges
incident to the growing set are ever examined, each undirected edge at most
once, with the per-edge Bernoulli draws vectorised over the whole frontier --
the subset-sampling idea behind modern RR-set generators (Guo et al., SIGMOD
2020 / TODS 2022) in array form.  A visited-stamp array avoids O(n) resets, so
per-set cost is proportional to the edges incident to the realised set, not to
the graph.
"""

import numpy as np


def m_rr(inst, n_sets=200, seed=999):
    """Ranking score: per-node containment frequency over n_sets RR sets."""
    rng = np.random.default_rng(seed)
    indptr, indices, probs = inst.P.indptr, inst.P.indices, inst.P.data
    n = inst.n
    hits = np.zeros(n)
    stamp = np.full(n, -1, dtype=np.int64)
    roots = rng.choice(n, size=n_sets, p=inst.x)
    for t in range(n_sets):
        root = roots[t]
        stamp[root] = t
        reached = [np.array([root])]
        frontier = np.array([root])
        while frontier.size:
            starts = indptr[frontier]
            counts = indptr[frontier + 1] - starts
            total = int(counts.sum())
            if total == 0:
                break
            # flat gather of all adjacency slices of the frontier
            offs = np.repeat(np.cumsum(counts) - counts, counts)
            flat = np.repeat(starts, counts) + (np.arange(total) - offs)
            targets = indices[flat]
            fresh = stamp[targets] != t
            if not fresh.any():
                break
            targets = targets[fresh]
            live = rng.random(targets.size) < probs[flat[fresh]]
            if not live.any():
                break
            new = np.unique(targets[live])
            new = new[stamp[new] != t]          # dedupe across frontier nodes
            if new.size == 0:
                break
            stamp[new] = t
            reached.append(new)
            frontier = new
        hits[np.concatenate(reached)] += 1.0
    return hits / n_sets
