import json
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from escape_lib import (make_instance, ground_truth, rho, timed,
                        alpha_from_rule, diffuse, m_pesc, m_sum_p, m_irie,
                        m_mia, m_mc, m_escape_pull, m_ppr_push)

CAL = json.load(open("calibration.json"))
COEF = (CAL["intercept"], CAL["slope"])
AT = json.load(open("alpha_transfer.json"))
PUSH_COEF = (AT["rules"]["PPR push"][0], AT["rules"]["PPR push"][1])

# ---------------------------------------------------------------- identity
print("=== 1. identity check ===")
inst = make_instance("er", 60, 0.10, 0.30, 7)

# (a) live-edge / IC simulation: activate seed u, spread along live edges
def ic_spread(inst, u, n_sim, rng):
    tot = 0.0
    for _ in range(n_sim):
        keep = rng.random(inst.m) < inst.p
        ei, ej = inst.ei[keep], inst.ej[keep]
        A = sp.coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(inst.n, inst.n))
        _, lab = connected_components(A, directed=False)
        tot += inst.x[lab == lab[u]].sum()
    return tot / n_sim

rng = np.random.default_rng(0)
gt = ground_truth(inst, 40000, seed=3)
ic = np.array([ic_spread(inst, u, 4000, rng) for u in range(0, inst.n, 6)])
sel = np.arange(0, inst.n, 6)
print(f"  max |y* - IC single-seed weighted spread| = "
      f"{np.abs(gt[sel] - ic).max():.4f}  (MC noise ~ 1/sqrt(4000) = "
      f"{1/np.sqrt(4000):.4f})")

# (b) EPC: uniform prior, uniform p -> y* proportional to EPC score
inst_u = make_instance("er", 60, 0.25, 0.25, 7, prior_kind="uniform")
gt_u = ground_truth(inst_u, 40000, seed=4)
epc = gt_u * inst_u.n          # EPC counts nodes, y* averages unit mass
print(f"  EPC(v) = n * y*_v with uniform x and constant p: "
      f"mean EPC = {epc.mean():.3f}, matches by construction "
      f"(Spearman {rho(epc, gt_u):.6f})")

# ---------------------------------------------------------------- scale
print("\n=== 2. cost/accuracy at n=900 ===")
SC_TOPOS = ["er", "ba", "grid", "ws"]
SC_RANGES = [(0.02, 0.08), (0.08, 0.18), (0.28, 0.52)]
SC_SEEDS = [201, 202]
METHODS = {
    "P_esc alone":     lambda i: m_pesc(i),
    "sum_j p_ij":      lambda i: m_sum_p(i),
    "MIA/PMIA":        lambda i: m_mia(i, 1e-3),
    "IRIE":            lambda i: m_irie(i, 0.7),
    "PPR push rule-a": lambda i: diffuse(i, float(np.clip(
        PUSH_COEF[0] + PUSH_COEF[1] * i.mean_pesc, .4, 1)),
        use_pesc=False, orientation="push"),
    "escape rule-a":   lambda i: m_escape_pull(i, alpha_from_rule(i, COEF)),
    "MC 50":           lambda i: m_mc(i, 50),
    "MC 200":          lambda i: m_mc(i, 200),
    "MC 1000":         lambda i: m_mc(i, 1000),
}
acc = {k: [] for k in METHODS}
tim = {k: [] for k in METHODS}
for topo in SC_TOPOS:
    for (lo, hi) in SC_RANGES:
        for sd in SC_SEEDS:
            ins = make_instance(topo, 900, lo, hi, sd)
            g = ground_truth(ins, 4000, seed=7000 + sd)
            for nm, fn in METHODS.items():
                s, dt = timed(fn, ins)
                acc[nm].append(rho(s, g))
                tim[nm].append(dt * 1000)
    print("  scale", topo, flush=True)

print(f"\n  {'method':18s} {'mean rho':>9s} {'ms':>9s}")
scale_summary = {}
for nm in sorted(METHODS, key=lambda k: np.mean(acc[k])):
    scale_summary[nm] = dict(rho=float(np.mean(acc[nm])),
                             ms=float(np.mean(tim[nm])))
    print(f"  {nm:18s} {np.mean(acc[nm]):9.3f} {np.mean(tim[nm]):9.2f}")

# ------------------------------------------------- MC budget to match
print("\n=== 3. Monte-Carlo budget needed to match the heuristic (n=100) ===")
budgets = [5, 10, 20, 40, 80, 160, 320]
rows = []
for topo in ["er", "ba", "grid", "ws"]:
    for (lo, hi) in [(0.02, 0.08), (0.18, 0.38), (0.45, 0.72)]:
        for sd in [301, 302]:
            ins = make_instance(topo, 100, lo, hi, sd)
            g = ground_truth(ins, 3000, seed=9000 + sd)
            r_ours, t_ours = timed(m_escape_pull, ins,
                                   alpha=alpha_from_rule(ins, COEF))
            row = dict(topo=topo, ours=rho(r_ours, g), ours_ms=t_ours * 1000)
            for b in budgets:
                s, dt = timed(m_mc, ins, n_samples=b)
                row[f"mc{b}"] = rho(s, g)
                row[f"mc{b}_ms"] = dt * 1000
            rows.append(row)
print(f"  {'budget':>8s} {'mean rho':>9s} {'mean ms':>9s}")
print(f"  {'ours':>8s} {np.mean([r['ours'] for r in rows]):9.3f} "
      f"{np.mean([r['ours_ms'] for r in rows]):9.3f}")
mc_curve = {}
for b in budgets:
    mc_curve[b] = dict(rho=float(np.mean([r[f"mc{b}"] for r in rows])),
                       ms=float(np.mean([r[f"mc{b}_ms"] for r in rows])))
    print(f"  {b:8d} {mc_curve[b]['rho']:9.3f} {mc_curve[b]['ms']:9.3f}")

json.dump(dict(scale=scale_summary, mc_curve=mc_curve,
               ours_small=dict(rho=float(np.mean([r['ours'] for r in rows])),
                               ms=float(np.mean([r['ours_ms'] for r in rows])))),
          open("scale_identity.json", "w"), indent=1)
print("\nwrote scale_identity.json")