import json
import numpy as np
from scipy.stats import wilcoxon

# ---------------------------------------------------------------- data
big = [json.loads(l) for l in open("followup_n900.jsonl") if l.strip()]
big.sort(key=lambda z: z["id"])
M900 = ["P_esc alone", "sum_j p_ij", "MIA/PMIA", "IRIE",
        "push PPR rule-a", "escape rule-a", "MC 50", "MC 200"]

HO = json.load(open("holdout.json"))["records"]
NAME100 = {"P_esc alone": "P_esc alone", "sum_j p_ij": "sum_j p_ij",
           "MIA/PMIA": "MIA/PMIA (KDD10)", "IRIE": "IRIE (ICDM12)",
           "push PPR rule-a": "PPR push rule-a", "escape rule-a": "THIS WORK",
           "MC 50": "MC 50 samples", "MC 200": "MC 200 samples"}

small = [dict(mean_pesc=r["mean_pesc"],
              rho={k: r["rho"][v] for k, v in NAME100.items()},
              ms={k: r["ms"][v] for k, v in NAME100.items()}) for r in HO]
for r in big:
    r["_pe"] = r["mean_pesc"]
for r in small:
    r["_pe"] = r["mean_pesc"]


def fit_router(rows, edges, methods):
    """Best method per Pesc bucket, learned on `rows`."""
    pick = {}
    for b in range(len(edges) - 1):
        sel = [x for x in rows if edges[b] <= x["_pe"] < edges[b + 1]]
        if not sel:
            pick[b] = methods[0]
            continue
        pick[b] = max(methods,
                      key=lambda m: np.mean([x["rho"][m] for x in sel]))
    return pick


def apply_router(rows, edges, pick, field="rho"):
    out = []
    for x in rows:
        b = min(np.searchsorted(edges, x["_pe"], side="right") - 1,
                len(edges) - 2)
        b = max(b, 0)
        out.append(x[field][pick[b]])
    return np.array(out)


def kfold(rows, edges, methods, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    folds = np.array_split(idx, k)
    acc, cost = [], []
    for f in folds:
        te = [rows[i] for i in f]
        tr = [rows[i] for i in idx if i not in set(f.tolist())]
        pick = fit_router(tr, edges, methods)
        acc.append(apply_router(te, edges, pick))
        cost.append(apply_router(te, edges, pick, field="ms"))
    return np.concatenate(acc), np.concatenate(cost)


E8 = np.linspace(0, 1, 9)

print("=" * 74)
print("1. k-FOLD CROSS-VALIDATION, n=900, 400 instances, 8 buckets")
print("=" * 74)
r_acc, r_cost = kfold(big, E8, M900, k=5)
print(f"  {'method':22s} {'mean rho':>9s} {'mean ms':>9s}")
singles = {m: np.mean([x["rho"][m] for x in big]) for m in M900}
for m in sorted(M900, key=lambda z: -singles[z]):
    print(f"  {m:22s} {singles[m]:9.4f} "
          f"{np.mean([x['ms'][m] for x in big]):9.2f}")
print(f"  {'ROUTER (5-fold CV)':22s} {r_acc.mean():9.4f} {r_cost.mean():9.2f}")
bestm = max(singles, key=singles.get)
d = r_acc - np.array([x["rho"][bestm] for x in big])
print(f"\n  gain over best single ({bestm}): {r_acc.mean()-singles[bestm]:+.4f}"
      f"  (p={wilcoxon(d[np.abs(d)>1e-12]).pvalue:.2e})")
ceil = np.mean([max(x["rho"][m] for m in M900) for x in big])
print(f"  per-instance oracle SELECTION (ceiling): {ceil:.4f}")
print(f"  router captures {100*(r_acc.mean()-singles[bestm])/(ceil-singles[bestm]):.0f}%"
      f" of the available selection gain")

print("\n" + "=" * 74)
print("2. HOW FEW THRESHOLDS ARE NEEDED?  (5-fold CV each)")
print("=" * 74)
for name, edges in [("1 bucket (no routing)", np.array([0., 1.])),
                    ("2 buckets @0.50", np.array([0., .5, 1.])),
                    ("2 buckets @0.875", np.array([0., .875, 1.])),
                    ("3 buckets @0.625/0.875", np.array([0., .625, .875, 1.])),
                    ("4 buckets", np.linspace(0, 1, 5)),
                    ("8 buckets", E8),
                    ("16 buckets", np.linspace(0, 1, 17))]:
    a, c = kfold(big, edges, M900, k=5)
    print(f"  {name:26s} rho {a.mean():.4f}   ms {c.mean():6.2f}")

print("\n" + "=" * 74)
print("3. CROSS-SCALE TRANSFER OF THE THRESHOLDS")
print("=" * 74)
E3 = np.array([0., .625, .875, 1.])
pick900 = fit_router(big, E3, M900)
pick100 = fit_router(small, E3, M900)
print(f"  learned at n=900 : {[pick900[b] for b in range(3)]}")
print(f"  learned at n=100 : {[pick100[b] for b in range(3)]}")
a_cross = apply_router(small, E3, pick900)
a_native = apply_router(small, E3, pick100)
s100 = {m: np.mean([x["rho"][m] for x in small]) for m in M900}
b100 = max(s100, key=s100.get)
print(f"\n  at n=100, best single ({b100}): {s100[b100]:.4f}")
print(f"  at n=100, router fitted at n=100     : {a_native.mean():.4f}")
print(f"  at n=100, router fitted at n=900     : {a_cross.mean():.4f}  "
      f"<-- transfers without refitting")

print("\n" + "=" * 74)
print("4. WHAT THE ROUTER BUYS, BY STRATUM (n=900, 3-bucket rule)")
print("=" * 74)
print(f"  {'Pesc range':16s} {'chosen':16s} {'rho':>7s} {'ms':>8s} "
      f"{'vs IRIE':>9s}")
for b in range(3):
    sel = [x for x in big if E3[b] <= x["_pe"] < E3[b + 1]]
    m = pick900[b]
    rr = np.mean([x["rho"][m] for x in sel])
    mm = np.mean([x["ms"][m] for x in sel])
    ir = np.mean([x["rho"]["IRIE"] for x in sel])
    print(f"  [{E3[b]:.3f},{E3[b+1]:.3f})   {m:16s} {rr:7.4f} {mm:8.2f} "
          f"{rr-ir:+9.4f}")

json.dump(dict(kfold_rho=float(r_acc.mean()), kfold_ms=float(r_cost.mean()),
               best_single=bestm, best_single_rho=float(singles[bestm]),
               ceiling=float(ceil),
               pick3={str(k): v for k, v in pick900.items()},
               cross_scale=dict(n100_best=float(s100[b100]),
                                n100_router_native=float(a_native.mean()),
                                n100_router_from900=float(a_cross.mean()))),
          open("router.json", "w"), indent=1)
print("\nwrote router.json")