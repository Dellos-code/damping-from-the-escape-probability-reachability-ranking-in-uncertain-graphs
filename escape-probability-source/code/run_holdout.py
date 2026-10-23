import json
CAL = json.load(open("calibration.json")); COEF = (CAL["intercept"], CAL["slope"])
import numpy as np
from scipy.stats import wilcoxon
from escape_lib import (make_instance, ground_truth, rho, timed, oracle_alpha,
                        alpha_from_rule, m_prior, m_sum_p, m_pesc,
                        m_degree_discount, m_ppr_pull, m_ppr_push,
                        m_pull_times_pesc, m_escape_pull, m_irie, m_irie_wc,
                        m_mia, m_mc, m_ris)


TOPOS = ["er", "ba", "grid", "ws"]
RANGES = [(0.02, 0.08), (0.08, 0.18), (0.18, 0.38), (0.28, 0.52), (0.45, 0.72)]
SEEDS = [101, 102, 103, 104, 105]
N = 100
MC = 3000

METHODS = {
    "x only":                    lambda i: m_prior(i),
    "sum_j p_ij":                lambda i: m_sum_p(i),
    "degree discount (KDD09)":   lambda i: m_degree_discount(i),
    "P_esc alone":               lambda i: m_pesc(i),
    "PPR pull a=.85":            lambda i: m_ppr_pull(i, 0.85),
    "PPR pull rule-a":           lambda i: m_ppr_pull(i, alpha_from_rule(i, COEF)),
    "PPR push a=.85":            lambda i: m_ppr_push(i, 0.85),
    "PPR push rule-a":           lambda i: m_ppr_push(i, alpha_from_rule(i, COEF)),
    "pull x P_esc post hoc":     lambda i: m_pull_times_pesc(i, alpha_from_rule(i, COEF)),
    "escape-scaled a=1":         lambda i: m_escape_pull(i, 1.0),
    "IRIE (ICDM12)":             lambda i: m_irie(i, 0.7),
    "IRIE row-normalised":       lambda i: m_irie_wc(i, 0.7),
    "MIA/PMIA (KDD10)":          lambda i: m_mia(i, 1e-3),
    "MC 10 samples":             lambda i: m_mc(i, 10),
    "MC 50 samples":             lambda i: m_mc(i, 50),
    "MC 200 samples":            lambda i: m_mc(i, 200),
    "RIS 50 samples":            lambda i: m_ris(i, 50),
    "THIS WORK":                 lambda i: m_escape_pull(i, alpha_from_rule(i, COEF)),
}

records = []
for topo in TOPOS:
    for (lo, hi) in RANGES:
        for sd in SEEDS:
            inst = make_instance(topo, N, lo, hi, sd)
            gt = ground_truth(inst, MC, seed=5000 + sd)
            rec = dict(topology=topo, lo=lo, hi=hi, seed=sd,
                       mean_pesc=inst.mean_pesc,
                       alpha_rule=alpha_from_rule(inst, COEF), rho={}, ms={})
            for name, fn in METHODS.items():
                s, dt = timed(fn, inst)
                rec["rho"][name] = rho(s, gt)
                rec["ms"][name] = dt * 1000
            ao, ro = oracle_alpha(inst, gt)
            rec["rho"]["oracle a"] = ro
            rec["ms"]["oracle a"] = float("nan")
            rec["alpha_oracle"] = ao
            records.append(rec)
        print(f"done {topo} p[{lo:.2f},{hi:.2f}]", flush=True)

names = list(METHODS.keys()) + ["oracle a"]
M = {nm: np.array([r["rho"][nm] for r in records]) for nm in names}
T = {nm: np.array([r["ms"][nm] for r in records]) for nm in names}
ours = M["THIS WORK"]

print("\n" + "=" * 78)
print(f"{'method':28s} {'mean rho':>9s} {'gain':>8s} {'wins':>6s} "
      f"{'p':>10s} {'ms':>8s}")
print("=" * 78)
summary = []
for nm in sorted(names, key=lambda k: M[k].mean()):
    v = M[nm]
    if nm == "THIS WORK":
        line = (nm, v.mean(), None, None, None, T[nm].mean())
    else:
        d = ours - v
        nz = d[np.abs(d) > 1e-12]
        p = wilcoxon(nz).pvalue if len(nz) > 0 else 1.0
        line = (nm, v.mean(), v.mean() - ours.mean(), int((d > 0).sum()), p,
                T[nm].mean())
    summary.append(line)
    g = "" if line[2] is None else f"{line[2]:+.3f}"
    w = "" if line[3] is None else f"{line[3]}/100"
    pp = "" if line[4] is None else f"{line[4]:.1e}"
    print(f"{nm:28s} {line[1]:9.3f} {g:>8s} {w:>6s} {pp:>10s} {line[5]:8.3f}")

json.dump(dict(records=records, coef=COEF), open("holdout.json", "w"))
print("\nwrote holdout.json")