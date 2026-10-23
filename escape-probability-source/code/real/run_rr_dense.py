"""RR-set baseline on the 30 dense real instances (same configs, same GT)."""
import json
import sys
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "..")
from escape_lib import Instance, ground_truth, rho, timed, m_pesc, m_mc
from rr_ranking import m_rr
from scale_real import load_graph, bfs_ball, edges_of
from dense_panel import tune, TARGET, CFGS

import os
done = set()
if os.path.exists("rr_dense.jsonl"):
    done = {json.loads(l)["cfg_id"] for l in open("rr_dense.jsonl") if l.strip()}
out = open("rr_dense.jsonl", "a")
cache = {}
for c in CFGS:
    if c["id"] in done:
        continue
    if c["genre"] not in cache:
        cache.clear()
        cache[c["genre"]] = load_graph(c["genre"])
    G = cache[c["genre"]]
    if G.number_of_nodes() < c["n_t"]:
        continue
    S = bfs_ball(G, c["n_t"], seed=c["seed"])
    n = S.number_of_nodes()
    if n <= 100:
        continue                       # degenerate snowball samples, as before
    ei, ej = edges_of(S)
    rng = np.random.default_rng(c["seed"] + 5)
    x = rng.dirichlet(np.full(n, 0.7))
    x /= x.sum()
    p = tune(n, ei, ej, x, TARGET, c["seed"])
    inst = Instance(n, ei, ej, p, x)
    gt = ground_truth(inst, 3000, seed=c["seed"] * 7 + 1)
    rec = dict(cfg_id=c["id"], genre=c["genre"], n=n, m=int(inst.m),
               mean_pesc=float(inst.mean_pesc), rho={}, ms={})
    for nm, fn, kw in [("P_esc alone", m_pesc, {}),
                       ("MC 200", m_mc, dict(n_samples=200)),
                       ("RR 200", m_rr, dict(n_sets=200, seed=c["seed"] + 31)),
                       ("RR 800", m_rr, dict(n_sets=800, seed=c["seed"] + 63))]:
        s, dt = timed(fn, inst, **kw)
        rec["rho"][nm] = rho(s, gt)
        rec["ms"][nm] = dt * 1000
    out.write(json.dumps(rec) + "\n"); out.flush()
    print(f"{c['genre']:17s} n={n:6d} Pesc={inst.mean_pesc:.3f} | "
          f"Pesc {rec['rho']['P_esc alone']:.4f} "
          f"MC200 {rec['rho']['MC 200']:.4f} ({rec['ms']['MC 200']:.0f}ms) "
          f"RR200 {rec['rho']['RR 200']:.4f} ({rec['ms']['RR 200']:.0f}ms) "
          f"RR800 {rec['rho']['RR 800']:.4f}", flush=True)

out.close()
rows = [json.loads(l) for l in open("rr_dense.jsonl") if l.strip()]
json.dump(rows, open("rr_dense.json", "w"), indent=1)
print(f"\n{len(rows)} instances")
print(f"{'method':12s} {'mean rho':>9s} {'mean ms':>9s}")
for m in ["P_esc alone", "MC 200", "RR 200", "RR 800"]:
    print(f"{m:12s} {np.mean([r['rho'][m] for r in rows]):9.4f} "
          f"{np.mean([r['ms'][m] for r in rows]):9.1f}")
pe = np.array([r["rho"]["P_esc alone"] for r in rows])
mc = np.array([r["rho"]["MC 200"] for r in rows])
for m, base, bn in [("RR 200", mc, "MC 200"), ("RR 800", mc, "MC 200"),
                    ("RR 200", pe, "Pesc"), ("RR 800", pe, "Pesc")]:
    d = np.array([r["rho"][m] for r in rows]) - base
    nz = d[np.abs(d) > 1e-12]
    print(f"{m} vs {bn}: {d.mean():+.4f}  wins {(d>0).sum()}/{len(rows)}  "
          f"p={wilcoxon(nz).pvalue:.2e}")
