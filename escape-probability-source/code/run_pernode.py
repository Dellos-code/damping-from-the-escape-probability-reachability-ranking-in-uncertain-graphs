import json
import numpy as np
from scipy.stats import wilcoxon
from escape_lib import (make_instance, ground_truth, rho, alpha_from_rule,
                        m_escape_pull, oracle_alpha)

CAL = json.load(open("calibration.json"))
COEF = (CAL["intercept"], CAL["slope"])


# ---------------------------------------------------------------- machinery
def smoothed_pesc(inst, beta, weighting="log", iters=200, tol=1e-13):
    """S = (1-beta) Pesc + beta W S : escape probability diffused over
    the neighbourhood, decaying with distance at rate beta."""
    if beta <= 0:
        return inst.pesc.copy()
    W = inst.W(weighting)
    pe = inst.pesc
    S = pe.copy()
    for _ in range(iters):
        Sn = (1.0 - beta) * pe + beta * (W @ S)
        if np.abs(Sn - S).max() < tol:
            return Sn
        S = Sn
    return S


def diffuse_vec(inst, a, weighting="log", tol=1e-13, max_iter=800):
    """Escape-scaled pull iteration with a per-node damping VECTOR a."""
    W = inst.W(weighting)
    pe = inst.pesc
    x = inst.x
    a = np.asarray(a, dtype=float)
    y = x.copy()
    for _ in range(max_iter):
        z = pe * (W @ y)
        yn = (1.0 - a) * x + a * z
        s = yn.sum()
        if s > 0:
            yn = yn / s
        if np.abs(yn - y).max() < tol:
            return yn
        y = yn
    return y


# =================================================== STEP 1: the ceiling
print("=" * 76)
print("STEP 1  CEILING: can a per-node alpha beat the scalar oracle at all?")
print("=" * 76)

TOPOS = ["er", "ba", "grid", "ws"]
RANGES = [(0.02, 0.08), (0.18, 0.38), (0.45, 0.72)]
SEEDS = [101, 102]
CAND = np.round(np.arange(0.40, 1.001, 0.075), 4)

sc_or, pn_or, sc_rule = [], [], []
for topo in TOPOS:
    for (lo, hi) in RANGES:
        for sd in SEEDS:
            ins = make_instance(topo, 100, lo, hi, sd)
            gt = ground_truth(ins, 3000, seed=5000 + sd)

            a_scalar, r_scalar = oracle_alpha(ins, gt)
            sc_or.append(r_scalar)
            sc_rule.append(rho(m_escape_pull(ins, alpha=alpha_from_rule(ins, COEF)), gt))

            # coordinate ascent on the alpha vector, warm-started at the
            # scalar oracle; 3 sweeps over all vertices
            a = np.full(ins.n, a_scalar)
            best = rho(diffuse_vec(ins, a), gt)
            for _ in range(3):
                improved = False
                for i in range(ins.n):
                    keep, cur = a[i], best
                    for c in CAND:
                        if c == keep:
                            continue
                        a[i] = c
                        r = rho(diffuse_vec(ins, a), gt)
                        if r > cur + 1e-9:
                            cur, keep = r, c
                    a[i] = keep
                    if cur > best + 1e-9:
                        best, improved = cur, True
                if not improved:
                    break
            pn_or.append(best)
    print("  ceiling swept", topo, flush=True)

sc_or, pn_or, sc_rule = map(np.array, (sc_or, pn_or, sc_rule))
head = pn_or - sc_or
print(f"\n  instances                       : {len(sc_or)}")
print(f"  scalar rule  (what we ship)     : {sc_rule.mean():.4f}")
print(f"  scalar ORACLE                   : {sc_or.mean():.4f}")
print(f"  per-node ORACLE (coord. ascent) : {pn_or.mean():.4f}")
print(f"  headroom over scalar oracle     : {head.mean():+.4f} "
      f"(max {head.max():+.4f}, wins {(head>1e-6).sum()}/{len(head)})")
print(f"  headroom over the shipped rule  : {(pn_or-sc_rule).mean():+.4f}")

# =================================================== STEP 2: rule families
print("\n" + "=" * 76)
print("STEP 2  RULE FAMILIES: fit c0,c1 per family on the calibration set")
print("=" * 76)

CAL_TOPOS = ["er", "ba", "grid"]
CAL_RANGES = [(0.01, 0.05), (0.03, 0.10), (0.07, 0.20),
              (0.15, 0.35), (0.25, 0.50), (0.40, 0.70)]
CAL_SEEDS = [11, 12, 13]
cal = []
for topo in CAL_TOPOS:
    for (lo, hi) in CAL_RANGES:
        for sd in CAL_SEEDS:
            ins = make_instance(topo, 64, lo, hi, sd)
            cal.append((ins, ground_truth(ins, 2500, seed=1000 + sd)))
print(f"  calibration instances: {len(cal)}")

BETAS = [None, 0.0, 0.25, 0.5, 0.75, 0.9]     # None = global mean (current)
C0 = np.round(np.arange(0.20, 0.91, 0.10), 3)
C1 = np.round(np.arange(0.00, 1.01, 0.10), 3)


def stat_for(inst, beta):
    if beta is None:
        return np.full(inst.n, inst.mean_pesc)
    return smoothed_pesc(inst, beta)


fitted = {}
for beta in BETAS:
    pre = [(i, g, stat_for(i, beta)) for i, g in cal]
    best, bc = -2.0, None
    for c0 in C0:
        for c1 in C1:
            tot = 0.0
            for i, g, S in pre:
                a = np.clip(c0 + c1 * S, 0.4, 1.0)
                tot += rho(diffuse_vec(i, a), g)
            m = tot / len(pre)
            if m > best:
                best, bc = m, (float(c0), float(c1))
    fitted[str(beta)] = dict(c0=bc[0], c1=bc[1], cal_rho=float(best))
    lab = "global mean" if beta is None else f"beta={beta}"
    print(f"  {lab:14s} -> a=clip({bc[0]:.2f}+{bc[1]:.2f}*S)  "
          f"calibration rho {best:.4f}")

# ---- evaluate all families out of sample
print("\n" + "=" * 76)
print("HELD-OUT, n=100, 100 instances")
print("=" * 76)
HO = json.load(open("holdout.json"))
insts = []
for r in HO["records"]:
    ins = make_instance(r["topology"], 100, r["lo"], r["hi"], r["seed"])
    insts.append((ins, ground_truth(ins, 3000, seed=5000 + r["seed"])))

res = {}
for beta in BETAS:
    c0, c1 = fitted[str(beta)]["c0"], fitted[str(beta)]["c1"]
    vals = []
    for ins, g in insts:
        a = np.clip(c0 + c1 * stat_for(ins, beta), 0.4, 1.0)
        vals.append(rho(diffuse_vec(ins, a), g))
    res[str(beta)] = np.array(vals)

ship = np.array([rho(m_escape_pull(i, alpha=alpha_from_rule(i, COEF)), g)
                 for i, g in insts])
res["shipped"] = ship

base = res["None"]
print(f"  {'family':22s} {'mean rho':>9s} {'vs global':>10s} {'wins':>9s} {'p':>10s}")
for key in ["shipped", "None", "0.0", "0.25", "0.5", "0.75", "0.9"]:
    v = res[key]
    lab = {"shipped": "shipped scalar rule", "None": "refit global mean",
           "0.0": "per-node, local Pesc"}.get(key, f"per-node, beta={key}")
    if key == "None":
        print(f"  {lab:22s} {v.mean():9.4f} {'--':>10s} {'--':>9s} {'--':>10s}")
        continue
    d = v - base
    nz = d[np.abs(d) > 1e-12]
    p = wilcoxon(nz).pvalue if len(nz) else 1.0
    print(f"  {lab:22s} {v.mean():9.4f} {d.mean():+10.4f} "
          f"{(d>0).sum():4d}/100 {p:10.2e}")

json.dump(dict(ceiling=dict(scalar_rule=float(sc_rule.mean()),
                            scalar_oracle=float(sc_or.mean()),
                            pernode_oracle=float(pn_or.mean()),
                            headroom=float(head.mean()),
                            n=len(sc_or)),
               fitted=fitted,
               heldout={k: float(v.mean()) for k, v in res.items()}),
          open("pernode.json", "w"), indent=1)
print("\nwrote pernode.json")