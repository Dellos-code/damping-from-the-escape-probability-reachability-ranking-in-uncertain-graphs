"""RR-set baseline on the synthetic hold-out panel (same instances, same GT)."""
import json
import numpy as np
from scipy.stats import wilcoxon
from escape_lib import (make_instance, ground_truth, rho, timed,
                        alpha_from_rule, m_escape_pull, m_mc, m_irie)
from rr_ranking import m_rr

CAL = json.load(open("calibration.json"))
COEF = (CAL["intercept"], CAL["slope"])
TOPOS = ["er", "ba", "grid", "ws"]
RANGES = [(0.02, 0.08), (0.08, 0.18), (0.18, 0.38), (0.28, 0.52), (0.45, 0.72)]
SEEDS = [101, 102, 103, 104, 105]

rows = []
for topo in TOPOS:
    for (lo, hi) in RANGES:
        for sd in SEEDS:
            inst = make_instance(topo, 100, lo, hi, sd)
            gt = ground_truth(inst, 3000, seed=5000 + sd)
            rec = dict(topo=topo, lo=lo, hi=hi, seed=sd,
                       mean_pesc=inst.mean_pesc, rho={}, ms={})
            for nm, fn, kw in [
                ("rule-a", m_escape_pull, dict(alpha=alpha_from_rule(inst, COEF))),
                ("IRIE", m_irie, dict(alpha=0.7)),
                ("MC 200", m_mc, dict(n_samples=200)),
                ("RR 200", m_rr, dict(n_sets=200, seed=77 + sd)),
                ("RR 800", m_rr, dict(n_sets=800, seed=177 + sd)),
                ("RR 3200", m_rr, dict(n_sets=3200, seed=277 + sd)),
            ]:
                s, dt = timed(fn, inst, **kw)
                rec["rho"][nm] = rho(s, gt)
                rec["ms"][nm] = dt * 1000
            rows.append(rec)
    print("done", topo, flush=True)

json.dump(rows, open("rr_holdout.json", "w"), indent=1)
M = ["rule-a", "IRIE", "MC 200", "RR 200", "RR 800", "RR 3200"]
print(f"\n{'method':10s} {'mean rho':>9s} {'mean ms':>9s}")
for m in M:
    print(f"{m:10s} {np.mean([r['rho'][m] for r in rows]):9.4f} "
          f"{np.mean([r['ms'][m] for r in rows]):9.2f}")
mc = np.array([r["rho"]["MC 200"] for r in rows])
for m in ["RR 200", "RR 800", "RR 3200"]:
    v = np.array([r["rho"][m] for r in rows]); d = v - mc
    nz = d[np.abs(d) > 1e-12]
    print(f"{m} vs MC 200: {d.mean():+.4f}  wins {(d>0).sum()}/100  "
          f"p={wilcoxon(nz).pvalue:.2e}")
