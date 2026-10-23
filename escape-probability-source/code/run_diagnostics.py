import json
import numpy as np
from scipy.stats import wilcoxon, pearsonr
from escape_lib import (make_instance, ground_truth, rho, diffuse,
                        alpha_from_rule, m_escape_pull)

CAL = json.load(open("calibration.json"))
COEF = (CAL["intercept"], CAL["slope"])
AT = json.load(open("alpha_transfer.json"))
PUSH_COEF = tuple(AT["rules"]["PPR push"])
GRID = np.round(np.arange(0.40, 1.001, 0.025), 4)

TOPOS = ["er", "ba", "grid", "ws"]
DEGS = [4, 6, 8, 10, 12]
PRIORS = ["dirichlet", "uniform", "sparse"]


def stratified_sample(n, per_bin, n_bins=8, seed=7):
    """Same design as the n=900 study, at arbitrary n."""
    rng = np.random.default_rng(seed)
    bins = [[] for _ in range(n_bins)]
    tries = 0
    while sum(len(b) for b in bins) < n_bins * per_bin and tries < 200000:
        tries += 1
        topo = TOPOS[rng.integers(len(TOPOS))]
        deg = int(DEGS[rng.integers(len(DEGS))])
        p_lo = float(np.exp(rng.uniform(np.log(0.005), np.log(0.65))))
        p_hi = float(min(p_lo + rng.uniform(0.02, 0.30), 0.97))
        prior = PRIORS[rng.integers(len(PRIORS))]
        sd = int(rng.integers(1, 10**6))
        inst = make_instance(topo, n, p_lo, p_hi, sd, prior_kind=prior,
                             avg_deg=deg)
        b = min(int(inst.mean_pesc * n_bins), n_bins - 1)
        if len(bins[b]) >= per_bin:
            continue
        bins[b].append((topo, deg, p_lo, p_hi, prior, sd))
    return [c for b in bins for c in b], tries


# ===================================================================== A
print("=" * 78)
print("A. SCALE vs DESIGN: the same stratified design, run at n=100")
print("=" * 78)
cfgs, tries = stratified_sample(100, 25)
print(f"   {len(cfgs)} instances (8 strata x 25), {tries} draws\n")

esc, push, pes = [], [], []
for (topo, deg, lo, hi, prior, sd) in cfgs:
    ins = make_instance(topo, 100, lo, hi, sd, prior_kind=prior, avg_deg=deg)
    gt = ground_truth(ins, 3000, seed=sd % 10**6 + 3)
    a_e = alpha_from_rule(ins, COEF)
    a_p = float(np.clip(PUSH_COEF[0] + PUSH_COEF[1] * ins.mean_pesc, 0.4, 1.0))
    esc.append(rho(diffuse(ins, a_e, use_pesc=True, orientation="pull"), gt))
    push.append(rho(diffuse(ins, a_p, use_pesc=False, orientation="push"), gt))
    pes.append(ins.mean_pesc)
esc, push = np.array(esc), np.array(push)
d = esc - push
nz = d[np.abs(d) > 1e-12]
print(f"   escape-scaled rule-a : {esc.mean():.4f}")
print(f"   push PPR   rule-a    : {push.mean():.4f}")
print(f"   difference           : {d.mean():+.4f}  "
      f"(escape wins {(d>0).sum()}/{len(d)}, p={wilcoxon(nz).pvalue:.2e})")
print(f"\n   for reference, grid design at n=100 : escape 0.894, push 0.904 "
      f"(p=0.60)")
print(f"   stratified design at n=900          : escape 0.910, push 0.857 "
      f"(p=1.5e-21)")
verdict = ("DESIGN, not scale" if wilcoxon(nz).pvalue < 0.05 and d.mean() > 0
           else "SCALE, not design")
print(f"\n   >>> the gap is driven by: {verdict}")

# ===================================================================== B
print("\n" + "=" * 78)
print("B. IS THE rho-vs-alpha OPTIMUM FLATTER AT n=900?")
print("=" * 78)


def plateau_width(curve, grid, eps=0.005):
    """Total width of alpha values within eps of the instance's own maximum."""
    c = np.asarray(curve)
    good = c >= c.max() - eps
    return float(good.sum() / len(good))          # fraction of the grid


# n = 64 : recompute curves on the calibration instances
CAL_TOPOS = ["er", "ba", "grid"]
CAL_RANGES = [(0.01, 0.05), (0.03, 0.10), (0.07, 0.20),
              (0.15, 0.35), (0.25, 0.50), (0.40, 0.70)]
w64 = []
for topo in CAL_TOPOS:
    for (lo, hi) in CAL_RANGES:
        for sd in [11, 12, 13]:
            ins = make_instance(topo, 64, lo, hi, sd)
            gt = ground_truth(ins, 2500, seed=1000 + sd)
            c = [rho(diffuse(ins, float(a), use_pesc=True, orientation="pull"),
                     gt) for a in GRID]
            w64.append(plateau_width(c, GRID))

fu = [json.loads(l) for l in open("followup_n900.jsonl") if l.strip()]
w900 = [plateau_width(r["curve_escape"], GRID) for r in fu]

print(f"   fraction of the alpha grid within 0.005 of the instance optimum")
print(f"     n=64  ({len(w64)} instances) : {np.mean(w64):.3f} "
      f"(median {np.median(w64):.3f})")
print(f"     n=900 ({len(w900)} instances): {np.mean(w900):.3f} "
      f"(median {np.median(w900):.3f})")
ratio = np.mean(w900) / np.mean(w64)
print(f"     ratio n=900 / n=64 : {ratio:.2f}x")
print(f"\n   >>> the 'flatter optimum' explanation is "
      f"{'SUPPORTED' if ratio > 1.15 else 'NOT SUPPORTED'}")

# spread of rho across the whole grid, as a second view
sp64 = np.mean([1.0] * 0) if False else None
print("\n   secondary view: how much rho is at stake across the grid")
print(f"     n=900 mean (max - min) over alpha : "
      f"{np.mean([max(r['curve_escape'])-min(r['curve_escape']) for r in fu]):.3f}")

# ===================================================================== C
print("\n" + "=" * 78)
print("C. SENSITIVITY OF THE RULE TO ITS COEFFICIENTS (held-out, n=100)")
print("=" * 78)
HO = json.load(open("holdout.json"))
base_cfgs = [(r["topology"], r["lo"], r["hi"], r["seed"]) for r in HO["records"]]
insts = []
for (topo, lo, hi, sd) in base_cfgs:
    ins = make_instance(topo, 100, lo, hi, sd)
    insts.append((ins, ground_truth(ins, 3000, seed=5000 + sd)))

a0, b0 = COEF
print(f"   fitted rule: clip({a0:.3f} + {b0:.3f} Pesc, 0.4, 1)")
print(f"   {'perturbation':28s} {'mean rho':>9s} {'delta':>8s}")
base = np.mean([rho(m_escape_pull(i, alpha=float(np.clip(a0 + b0 * i.mean_pesc,
                                                         .4, 1))), g)
                for i, g in insts])
print(f"   {'(none)':28s} {base:9.4f} {0.0:+8.4f}")
sens = {}
for lbl, (da, db) in [("intercept -0.10", (-0.10, 0.0)),
                      ("intercept +0.10", (+0.10, 0.0)),
                      ("slope -0.20", (0.0, -0.20)),
                      ("slope +0.20", (0.0, +0.20)),
                      ("both -10% relative", (-0.1 * a0, -0.1 * b0)),
                      ("both +10% relative", (+0.1 * a0, +0.1 * b0)),
                      ("rounded to 0.5+0.6", (0.5 - a0, 0.6 - b0))]:
    aa, bb = a0 + da, b0 + db
    v = np.mean([rho(m_escape_pull(i, alpha=float(np.clip(aa + bb * i.mean_pesc,
                                                          .4, 1))), g)
                 for i, g in insts])
    sens[lbl] = float(v)
    print(f"   {lbl:28s} {v:9.4f} {v-base:+8.4f}")

json.dump(dict(scale_vs_design=dict(n100_strat_escape=float(esc.mean()),
                                    n100_strat_push=float(push.mean()),
                                    diff=float(d.mean()),
                                    wins=int((d > 0).sum()), n=len(d),
                                    p=float(wilcoxon(nz).pvalue)),
               plateau=dict(n64=float(np.mean(w64)), n900=float(np.mean(w900)),
                            ratio=float(ratio)),
               sensitivity=dict(base=float(base), **sens)),
          open("diagnostics.json", "w"), indent=1)
print("\nwrote diagnostics.json")