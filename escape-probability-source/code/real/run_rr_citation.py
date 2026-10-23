"""RR-set baseline on the 35 real citation instances (same configs, same GT)."""
import json
import pickle
import sys
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "..")
from escape_lib import ground_truth, rho, timed, m_mc
from rr_ranking import m_rr
from real_experiment import make_real_instance

topos = pickle.load(open("topologies.pkl", "rb"))
cfgs = json.load(open("real_configs.json"))
rows = []
for cfg in cfgs:
    ins = make_real_instance(topos, cfg)
    gt = ground_truth(ins, 3000, seed=cfg["seed"] + 7)
    rec = dict(cfg)
    rec.update(n=ins.n, m=int(ins.m), mean_pesc=float(ins.mean_pesc),
               rho={}, ms={})
    for nm, fn, kw in [("MC 50", m_mc, dict(n_samples=50)),
                       ("MC 200", m_mc, dict(n_samples=200)),
                       ("RR 200", m_rr, dict(n_sets=200, seed=cfg["seed"] + 31)),
                       ("RR 800", m_rr, dict(n_sets=800, seed=cfg["seed"] + 63))]:
        s, dt = timed(fn, ins, **kw)
        rec["rho"][nm] = rho(s, gt)
        rec["ms"][nm] = dt * 1000
    rows.append(rec)
    print(f"[{cfg['id']:2d}] {cfg['topo']:15s} Pesc={ins.mean_pesc:.3f} "
          f"MC200 {rec['rho']['MC 200']:.4f} ({rec['ms']['MC 200']:.0f}ms) "
          f"RR200 {rec['rho']['RR 200']:.4f} ({rec['ms']['RR 200']:.0f}ms) "
          f"RR800 {rec['rho']['RR 800']:.4f} ({rec['ms']['RR 800']:.0f}ms)",
          flush=True)

json.dump(rows, open("rr_citation.json", "w"), indent=1)
print(f"\n{'method':8s} {'mean rho':>9s} {'mean ms':>9s}")
for m in ["MC 50", "MC 200", "RR 200", "RR 800"]:
    print(f"{m:8s} {np.mean([r['rho'][m] for r in rows]):9.4f} "
          f"{np.mean([r['ms'][m] for r in rows]):9.1f}")
mc = np.array([r["rho"]["MC 200"] for r in rows])
for m in ["RR 200", "RR 800"]:
    d = np.array([r["rho"][m] for r in rows]) - mc
    print(f"{m} vs MC 200: {d.mean():+.4f}  wins {(d>0).sum()}/35  "
          f"p={wilcoxon(d[np.abs(d)>1e-12]).pvalue:.2e}")
