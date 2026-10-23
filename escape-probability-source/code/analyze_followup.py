import json
import numpy as np
from scipy.stats import wilcoxon, pearsonr, spearmanr

CFG = json.load(open("followup_configs.json"))
recs = [json.loads(l) for l in open("followup_n900.jsonl") if l.strip()]
recs.sort(key=lambda r: r["id"])
FU = dict(alpha_grid=CFG["alpha_grid"], coef_escape=CFG["coef_escape"],
          coef_push=CFG["coef_push"], n_bins=CFG["n_bins"],
          per_bin=CFG["per_bin"])
GRID = np.array(FU["alpha_grid"])
OLD = json.load(open("final.json"))          # the original 36-instance panel

METHODS = ["P_esc alone", "sum_j p_ij", "MIA/PMIA", "IRIE",
           "push PPR rule-a", "escape rule-a", "MC 50", "MC 200"]
REF = "escape rule-a"


def boot_ci(d, B=20000, seed=0):
    """Bootstrap 95% CI for the mean of paired differences."""
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


R = {m: np.array([x["rho"][m] for x in recs]) for m in METHODS}
T = {m: np.array([x["ms"][m] for x in recs]) for m in METHODS}
pe = np.array([x["mean_pesc"] for x in recs])
ref = R[REF]

print("=" * 96)
print(f"STRATIFIED-RANDOM PANEL, n=900, {len(recs)} instances "
      f"(8 Pesc strata x 50), vs {REF}")
print("=" * 96)
print(f"{'method':18s} {'mean rho':>9s} {'diff':>8s} {'95% CI':>18s} "
      f"{'wins':>9s} {'p':>10s} {'ms':>8s}")
rows = {}
for m in sorted(METHODS, key=lambda k: R[k].mean()):
    v = R[m]
    if m == REF:
        rows[m] = dict(rho=float(v.mean()), ms=float(T[m].mean()))
        print(f"{m:18s} {v.mean():9.4f} {'--':>8s} {'--':>18s} {'--':>9s} "
              f"{'--':>10s} {T[m].mean():8.2f}")
        continue
    d = ref - v
    nz = d[np.abs(d) > 1e-12]
    p = wilcoxon(nz).pvalue if len(nz) else 1.0
    lo, hi = boot_ci(d)
    rows[m] = dict(rho=float(v.mean()), diff=float(v.mean() - ref.mean()),
                   ci=[-hi, -lo], wins=int((d > 0).sum()), p=float(p),
                   ms=float(T[m].mean()))
    print(f"{m:18s} {v.mean():9.4f} {v.mean()-ref.mean():+8.4f} "
          f"[{-hi:+.4f},{-lo:+.4f}]".rjust(0).ljust(0).rjust(19)
          + f" {(d>0).sum():4d}/{len(d)} {p:10.2e} {T[m].mean():8.2f}")

# ---------------------------------------------- resolution of earlier nulls
print("\n" + "=" * 96)
print("DOES THE LARGER SAMPLE RESOLVE THE EARLIER NON-SIGNIFICANT COMPARISONS?")
print("=" * 96)
print(f"{'comparison':28s} {'old p (36)':>12s} {'new p':>12s} {'new diff':>10s} "
      f"{'verdict':>28s}")
oldmap = {"MIA/PMIA": "MIA/PMIA (KDD10)", "IRIE": "IRIE (ICDM12)",
          "push PPR rule-a": "PPR push rule-a", "MC 50": "MC 50",
          "MC 200": "MC 200", "P_esc alone": "P_esc alone",
          "sum_j p_ij": "sum_j p_ij"}
for m in ["MIA/PMIA", "IRIE", "push PPR rule-a", "MC 50", "MC 200"]:
    ok = OLD["n900"].get(oldmap[m], {})
    oldp = ok.get("p", float("nan"))
    newp = rows[m]["p"]
    if oldp > 0.05 and newp < 0.05:
        verd = "null RESOLVED -> real effect"
    elif oldp > 0.05 and newp >= 0.05:
        verd = "still null at 400"
    elif oldp <= 0.05 and newp < 0.05:
        verd = "confirmed"
    else:
        verd = "no longer significant"
    print(f"{REF+' vs '+m:28s} {oldp:12.3f} {newp:12.2e} "
          f"{rows[m]['diff']:+10.4f} {verd:>28s}")

# ---------------------------------------------- stratified breakdown
print("\n" + "=" * 96)
print("BY ESCAPE-PROBABILITY STRATUM (50 instances each)")
print("=" * 96)
show = ["P_esc alone", "MIA/PMIA", "IRIE", "push PPR rule-a",
        "escape rule-a", "MC 200"]
print(f"{'stratum':16s} " + " ".join(f"{s[:14]:>15s}" for s in show) + "   best")
strat = {}
for b in range(FU["n_bins"]):
    sel = np.array([x["bin"] == b for x in recs])
    if sel.sum() == 0:
        continue
    vals = {m: float(R[m][sel].mean()) for m in show}
    best = max(vals, key=vals.get)
    strat[b] = dict(vals=vals, n=int(sel.sum()), best=best)
    lab = f"[{b/FU['n_bins']:.2f},{(b+1)/FU['n_bins']:.2f})"
    print(f"{lab:16s} " + " ".join(f"{vals[m]:15.3f}" for m in show)
          + f"   {best}")

# ---------------------------------------------- does the rule hold at n=900?
print("\n" + "=" * 96)
print("SCALE TRANSFER: the alpha rule was fitted at n=64. Does it hold at n=900?")
print("=" * 96)
for fam, key, coef in [("escape-scaled", "alpha_opt_escape", FU["coef_escape"]),
                       ("push PPR", "alpha_opt_push", FU["coef_push"])]:
    ao = np.array([x[key] for x in recs])
    pr = pearsonr(pe, ao)
    sr = spearmanr(pe, ao)
    b900, a900 = np.polyfit(pe, ao, 1)
    a64, b64 = coef[0], coef[1]
    pred64 = np.clip(a64 + b64 * pe, 0.4, 1.0)
    rmse64 = float(np.sqrt(np.mean((pred64 - ao) ** 2)))
    rmse900 = float(np.sqrt(np.mean((np.clip(a900 + b900 * pe, .4, 1) - ao) ** 2)))
    print(f"\n{fam}:")
    print(f"  n=64  rule      : clip({a64:.3f} + {b64:.3f} Pesc)")
    print(f"  n=900 refit     : clip({a900:.3f} + {b900:.3f} Pesc)")
    print(f"  correlation at n=900 : Pearson {pr.statistic:+.3f} "
          f"(p={pr.pvalue:.1e}), Spearman {sr.statistic:+.3f}")
    print(f"  RMSE of n=64 rule at n=900 : {rmse64:.3f}   "
          f"(refit-at-900 RMSE {rmse900:.3f})")

# ---------------------------------------------- how much does the rule recover
print("\n" + "=" * 96)
print("RULE vs BEST FIXED alpha vs ORACLE, at n=900 (rule fitted at n=64)")
print("=" * 96)
for fam, ck, ok_, rk in [("escape-scaled", "curve_escape", "oracle_escape",
                          "escape rule-a"),
                         ("push PPR", "curve_push", "oracle_push",
                          "push PPR rule-a")]:
    curves = np.array([x[ck] for x in recs])
    mean_curve = curves.mean(axis=0)
    bf, ba = float(mean_curve.max()), float(GRID[int(mean_curve.argmax())])
    orc = float(np.mean([x[ok_] for x in recs]))
    rl = float(R[rk].mean())
    a85 = float(mean_curve[int(np.argmin(np.abs(GRID - 0.85)))])
    print(f"\n{fam}:")
    print(f"  fixed alpha=0.85              : {a85:.4f}")
    print(f"  best fixed alpha (hindsight)  : {bf:.4f}  at alpha={ba:.3f}")
    print(f"  rule-alpha (fitted at n=64)   : {rl:.4f}   "
          f"(+{rl-bf:.4f} over best fixed)")
    print(f"  per-instance oracle           : {orc:.4f}   "
          f"({100*rl/orc:.1f}% of oracle)")

out = dict(n=len(recs), rows=rows, strat=strat)
json.dump(out, open("followup_analysis.json", "w"), indent=1)
print("\nwrote followup_analysis.json")