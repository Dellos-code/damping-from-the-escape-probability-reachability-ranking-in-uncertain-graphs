import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from escape_lib import make_instance, ground_truth, oracle_alpha

CAL_TOPOS = ["er", "ba", "grid"]
CAL_RANGES = [(0.01, 0.05), (0.03, 0.10), (0.07, 0.20),
              (0.15, 0.35), (0.25, 0.50), (0.40, 0.70)]
CAL_SEEDS = [11, 12, 13]
N_CAL = 64
MC_CAL = 2500

rows = []
for topo in CAL_TOPOS:
    for (lo, hi) in CAL_RANGES:
        for sd in CAL_SEEDS:
            inst = make_instance(topo, N_CAL, lo, hi, sd)
            gt = ground_truth(inst, MC_CAL, seed=1000 + sd)
            a_opt, r_opt = oracle_alpha(inst, gt)
            rows.append(dict(topology=topo, lo=lo, hi=hi, seed=sd,
                             mean_pesc=inst.mean_pesc,
                             alpha_opt=a_opt, rho_opt=r_opt))
            print(f"{topo:5s} p[{lo:.2f},{hi:.2f}] s{sd}  "
                  f"Pesc={inst.mean_pesc:.3f}  a*={a_opt:.2f}  rho={r_opt:.3f}",
                  flush=True)

pe = np.array([r["mean_pesc"] for r in rows])
al = np.array([r["alpha_opt"] for r in rows])
b, a = np.polyfit(pe, al, 1)
pred = np.clip(a + b * pe, 0.4, 1.0)
rmse = float(np.sqrt(np.mean((pred - al) ** 2)))
pr = pearsonr(pe, al)
sr = spearmanr(pe, al)

print("\n=== CALIBRATION FIT ===")
print(f"instances       : {len(rows)}")
print(f"alpha = clip({a:.3f} + {b:.3f} * mean_Pesc, 0.4, 1.0)")
print(f"Pearson r       : {pr.statistic:+.3f}  (p={pr.pvalue:.2e})")
print(f"Spearman rho    : {sr.statistic:+.3f}")
print(f"residual RMSE   : {rmse:.3f}")

COEF = (float(a), float(b))
json.dump(dict(rows=rows, intercept=float(a), slope=float(b),
               pearson=float(pr.statistic), pearson_p=float(pr.pvalue),
               spearman=float(sr.statistic), rmse=rmse),
          open("calibration.json", "w"), indent=1)
print("\nwrote calibration.json")