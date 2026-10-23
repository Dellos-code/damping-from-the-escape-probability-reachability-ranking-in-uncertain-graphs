import json
import numpy as np
from scipy.stats import pearsonr
from escape_lib import (make_instance, ground_truth, rho, diffuse, m_irie_wc,
                        m_escape_pull, m_ppr_push)

CAL_TOPOS = ["er", "ba", "grid"]
CAL_RANGES = [(0.01, 0.05), (0.03, 0.10), (0.07, 0.20),
              (0.15, 0.35), (0.25, 0.50), (0.40, 0.70)]
CAL_SEEDS = [11, 12, 13]
GRID = np.round(np.arange(0.40, 1.001, 0.05), 3)

FAMILIES = {
    "escape-scaled pull": lambda i, a: diffuse(i, a, use_pesc=True,  orientation="pull"),
    "PPR push":           lambda i, a: diffuse(i, a, use_pesc=False, orientation="push"),
    "IRIE row-norm":      lambda i, a: m_irie_wc(i, alpha=a),
}

# ---------- calibrate each family separately ----------
cal = {k: {"pe": [], "a": []} for k in FAMILIES}
for topo in CAL_TOPOS:
    for (lo, hi) in CAL_RANGES:
        for sd in CAL_SEEDS:
            inst = make_instance(topo, 64, lo, hi, sd)
            gt = ground_truth(inst, 2500, seed=1000 + sd)
            for fam, fn in FAMILIES.items():
                best, ba = -2.0, 1.0
                for a in GRID:
                    r = rho(fn(inst, float(a)), gt)
                    if r > best:
                        best, ba = r, float(a)
                cal[fam]["pe"].append(inst.mean_pesc)
                cal[fam]["a"].append(ba)
    print("cal", topo, flush=True)

rules = {}
print("\n=== per-family alpha rules fitted on the SAME 54 instances ===")
for fam in FAMILIES:
    pe = np.array(cal[fam]["pe"])
    al = np.array(cal[fam]["a"])
    b, a = np.polyfit(pe, al, 1)
    r = pearsonr(pe, al)
    rules[fam] = (float(a), float(b))
    rm = float(np.sqrt(np.mean((np.clip(a + b * pe, .4, 1) - al) ** 2)))
    print(f"{fam:22s} alpha=clip({a:.3f}+{b:.3f}*Pesc,.4,1)  "
          f"r={r.statistic:+.3f}  RMSE={rm:.3f}")

# ---------- test out of sample ----------
TOPOS = ["er", "ba", "grid", "ws"]
RANGES = [(0.02, 0.08), (0.08, 0.18), (0.18, 0.38), (0.28, 0.52), (0.45, 0.72)]
SEEDS = [101, 102, 103, 104, 105]

res = {f"{fam}|{mode}": [] for fam in FAMILIES
       for mode in ("fixed .85", "rule", "oracle")}
for topo in TOPOS:
    for (lo, hi) in RANGES:
        for sd in SEEDS:
            inst = make_instance(topo, 100, lo, hi, sd)
            gt = ground_truth(inst, 3000, seed=5000 + sd)
            for fam, fn in FAMILIES.items():
                a0, b0 = rules[fam]
                ar = float(np.clip(a0 + b0 * inst.mean_pesc, 0.4, 1.0))
                res[f"{fam}|fixed .85"].append(rho(fn(inst, 0.85), gt))
                res[f"{fam}|rule"].append(rho(fn(inst, ar), gt))
                res[f"{fam}|oracle"].append(
                    max(rho(fn(inst, float(a)), gt) for a in GRID))
    print("test", topo, flush=True)

print("\n=== out-of-sample mean Spearman rho, 100 instances ===")
print(f"{'family':22s} {'fixed .85':>10s} {'rule-a':>10s} {'oracle':>10s} "
      f"{'gain':>8s} {'% oracle':>9s}")
out = {}
for fam in FAMILIES:
    f8 = np.mean(res[f"{fam}|fixed .85"])
    fr = np.mean(res[f"{fam}|rule"])
    fo = np.mean(res[f"{fam}|oracle"])
    out[fam] = dict(fixed=f8, rule=fr, oracle=fo,
                    intercept=rules[fam][0], slope=rules[fam][1])
    print(f"{fam:22s} {f8:10.3f} {fr:10.3f} {fo:10.3f} "
          f"{fr - f8:+8.3f} {100 * fr / fo:8.1f}%")

json.dump(dict(rules=rules, results={k: list(map(float, v))
                                     for k, v in res.items()}, summary=out),
          open("alpha_transfer.json", "w"), indent=1)
print("\nwrote alpha_transfer.json")