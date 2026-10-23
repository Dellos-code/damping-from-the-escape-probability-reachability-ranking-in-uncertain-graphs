"""
Where does the free score beat sampling, and is there a crossover in n?

Earlier evidence pulls in two directions.  On citation k-cores at n=1257-4201 in
the dense bucket, 200-sample Monte Carlo beat Pesc alone.  On coauthor-physics
at n=8000-32000 in the dense regime, Pesc alone beat it.  If both are right,
there is a crossover in n -- which would matter, because it is exactly the
regime the router dispatches to.

Design: four genres x five sizes x two BFS seeds, edge probability tuned per
instance so Pesc_bar hits a fixed dense target, so only n and genre vary.
Enough instances for a paired test rather than an anecdote.

Usage: python3 dense_panel.py <count>     (resumable, appends to dense_panel.jsonl)
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, "..")
from escape_lib import (Instance, ground_truth, rho, timed, m_pesc, m_sum_p,
                        m_irie, m_mc, m_escape_pull, alpha_from_rule)
from scale_real import load_graph, bfs_ball, edges_of, COEF

OUT = "dense_panel.jsonl"
GENRES = ["coauthor-physics", "coauthor-cs", "amazon-computers", "amazon-photo"]
SIZES = [1000, 2000, 4000, 8000, 16000]
SEEDS = [11, 23]
TARGET = 0.90

CFGS = [dict(id=i, genre=g, n_t=s, seed=sd)
        for i, (g, s, sd) in enumerate(
            (g, s, sd) for g in GENRES for s in SIZES for sd in SEEDS)]


def tune(n, ei, ej, x, target, seed):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.5, 1.5, len(ei))       # mild p-heterogeneity
    lo, hi = 1e-6, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        p = np.clip(base * mid, 1e-6, 0.999)
        if Instance(n, ei, ej, p, x).mean_pesc < target:
            lo = mid
        else:
            hi = mid
    return np.clip(base * (lo + hi) / 2, 1e-6, 0.999)


def main(count):
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [c for c in CFGS if c["id"] not in done][:count]
    print(f"{len(done)} done, {len(CFGS)-len(done)} pending; running {len(todo)}")
    cache = {}
    with open(OUT, "a") as fh:
        for cfg in todo:
            t0 = time.time()
            if cfg["genre"] not in cache:
                cache.clear()
                cache[cfg["genre"]] = load_graph(cfg["genre"])
            G = cache[cfg["genre"]]
            if G.number_of_nodes() < cfg["n_t"]:
                continue
            S = bfs_ball(G, cfg["n_t"], seed=cfg["seed"])
            n = S.number_of_nodes()
            ei, ej = edges_of(S)
            rng = np.random.default_rng(cfg["seed"] + 5)
            x = rng.dirichlet(np.full(n, 0.7))
            x /= x.sum()
            p = tune(n, ei, ej, x, TARGET, cfg["seed"])
            inst = Instance(n, ei, ej, p, x)
            gt = ground_truth(inst, 3000, seed=cfg["seed"] * 7 + 1)
            gt2 = ground_truth(inst, 3000, seed=cfg["seed"] * 13 + 3)
            a_e = alpha_from_rule(inst, COEF)
            METH = {"P_esc alone": lambda i: m_pesc(i),
                    "sum_j p_ij": lambda i: m_sum_p(i),
                    "IRIE": lambda i: m_irie(i, 0.7),
                    "escape rule-a": lambda i: m_escape_pull(i, a_e),
                    "MC 50": lambda i: m_mc(i, 50),
                    "MC 200": lambda i: m_mc(i, 200),
                    "MC 800": lambda i: m_mc(i, 800)}
            rec = dict(cfg)
            rec.update(n=n, m=int(inst.m), mean_pesc=float(inst.mean_pesc),
                       gt_self=float(rho(gt, gt2)), rho={}, ms={})
            for nm, fn in METH.items():
                s, dt = timed(fn, inst)
                rec["rho"][nm] = rho(s, gt)
                rec["ms"][nm] = dt * 1000
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  [{cfg['id']:2d}] {cfg['genre']:17s} n={n:6d} m={inst.m:7d} "
                  f"Pesc={inst.mean_pesc:.3f} | P_esc {rec['rho']['P_esc alone']:.4f} "
                  f"MC200 {rec['rho']['MC 200']:.4f} "
                  f"diff {rec['rho']['P_esc alone']-rec['rho']['MC 200']:+.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
