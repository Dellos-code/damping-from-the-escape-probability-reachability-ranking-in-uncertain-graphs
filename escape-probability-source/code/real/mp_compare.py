"""
The decisive comparison: message passing versus its own eta=1 limit.

Re-runs the 30 real dense-regime instances and adds
  * full message passing (iterated to convergence),
  * message passing truncated at k = 0,1,2,3 iterations (k=0 IS Pesc),
so we can see how much of the ranking signal the free zeroth iterate captures
and how quickly the remainder is recovered.
"""

import json
import sys
import time
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "..")
from escape_lib import Instance, ground_truth, rho, timed, m_pesc, m_mc
from mp_percolation import mp_percolation, m_mp_k
from scale_real import load_graph, bfs_ball, edges_of
from dense_panel import tune, TARGET

CFGS = [json.loads(l) for l in open("dense_panel.jsonl") if l.strip()]
CFGS = [c for c in CFGS if c["n"] > 100]
print(f"re-running {len(CFGS)} dense instances with message passing")

rows = []
cache = {}
t0 = time.time()
for c in CFGS:
    if c["genre"] not in cache:
        cache.clear()
        cache[c["genre"]] = load_graph(c["genre"])
    S = bfs_ball(cache[c["genre"]], c["n_t"], seed=c["seed"])
    n = S.number_of_nodes()
    ei, ej = edges_of(S)
    rng = np.random.default_rng(c["seed"] + 5)
    x = rng.dirichlet(np.full(n, 0.7))
    x /= x.sum()
    p = tune(n, ei, ej, x, TARGET, c["seed"])
    inst = Instance(n, ei, ej, p, x)
    gt = ground_truth(inst, 3000, seed=c["seed"] * 7 + 1)

    r = dict(genre=c["genre"], n=n, m=int(inst.m), mean_pesc=float(inst.mean_pesc),
             rho={}, ms={})
    s, dt = timed(m_pesc, inst)
    r["rho"]["Pesc (k=0)"] = rho(s, gt)
    r["ms"]["Pesc (k=0)"] = dt * 1000
    for k in (1, 2, 3):
        s, dt = timed(m_mp_k, inst, k=k)
        r["rho"][f"MP k={k}"] = rho(s, gt)
        r["ms"][f"MP k={k}"] = dt * 1000
    t1 = time.perf_counter()
    mu, iters = mp_percolation(inst, return_iters=True)
    r["rho"]["MP full"] = rho(mu, gt)
    r["ms"]["MP full"] = (time.perf_counter() - t1) * 1000
    r["mp_iters"] = int(iters)
    s, dt = timed(m_mc, inst, n_samples=200)
    r["rho"]["MC 200"] = rho(s, gt)
    r["ms"]["MC 200"] = dt * 1000
    rows.append(r)
    print(f"  {c['genre']:17s} n={n:6d} | Pesc {r['rho']['Pesc (k=0)']:.4f} "
          f"k1 {r['rho']['MP k=1']:.4f} k2 {r['rho']['MP k=2']:.4f} "
          f"full {r['rho']['MP full']:.4f} ({iters} it) "
          f"MC200 {r['rho']['MC 200']:.4f}", flush=True)

json.dump(rows, open("mp_dense.json", "w"), indent=1)

M = ["Pesc (k=0)", "MP k=1", "MP k=2", "MP k=3", "MP full", "MC 200"]
print("\n" + "=" * 66)
print(f"{'method':14s} {'mean rho':>9s} {'mean ms':>9s} {'vs Pesc':>10s} {'p':>10s}")
pe = np.array([r["rho"]["Pesc (k=0)"] for r in rows])
for m in M:
    v = np.array([r["rho"][m] for r in rows])
    t = np.mean([r["ms"][m] for r in rows])
    if m == "Pesc (k=0)":
        print(f"{m:14s} {v.mean():9.4f} {t:9.2f} {'--':>10s} {'--':>10s}")
        continue
    d = v - pe
    nz = d[np.abs(d) > 1e-12]
    p = wilcoxon(nz).pvalue if len(nz) else 1.0
    print(f"{m:14s} {v.mean():9.4f} {t:9.2f} {d.mean():+10.4f} {p:10.2e}")
print(f"\nmean MP iterations to convergence: {np.mean([r['mp_iters'] for r in rows]):.1f}")
mp = np.array([r["rho"]["MP full"] for r in rows])
mc = np.array([r["rho"]["MC 200"] for r in rows])
d = mp - mc
print(f"MP full vs MC 200: {d.mean():+.4f}, MP wins {(d>0).sum()}/{len(d)}, "
      f"p={wilcoxon(d[np.abs(d)>1e-12]).pvalue:.2e}")
print(f"cost ratio MC200 / MPfull : "
      f"{np.mean([r['ms']['MC 200'] for r in rows])/np.mean([r['ms']['MP full'] for r in rows]):.1f}x")
print(f"cost ratio MPfull / Pesc  : "
      f"{np.mean([r['ms']['MP full'] for r in rows])/max(np.mean([r['ms']['Pesc (k=0)'] for r in rows]),1e-6):.0f}x")
print(f"\ntotal runtime {(time.time()-t0)/60:.1f} min")
