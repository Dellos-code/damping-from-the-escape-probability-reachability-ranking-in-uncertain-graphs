import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAL = json.load(open("calibration.json"))
AT = json.load(open("alpha_transfer.json"))
HO = json.load(open("holdout.json"))
SI = json.load(open("scale_identity.json"))
SW = json.load(open("alpha_sweep.json"))

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 160})
fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
ax = axes.ravel()

# ---------------- A: calibration -------------------------------------------
pe = np.array([r["mean_pesc"] for r in CAL["rows"]])
al = np.array([r["alpha_opt"] for r in CAL["rows"]])
ax[0].scatter(pe, al, s=30, alpha=0.65, edgecolor="k", linewidth=0.4,
              zorder=3, label="optimal $\\alpha$ per training instance")
g = np.linspace(0, 1, 200)
a0, b0 = CAL["intercept"], CAL["slope"]
ap, bp = AT["rules"]["PPR push"]
ax[0].plot(g, np.clip(a0 + b0 * g, 0.4, 1.0), "r-", lw=2.2,
           label=rf"escape-scaled: ${a0:.3f}+{b0:.3f}\bar{{P}}_{{esc}}$"
                 f"\n   $r={CAL['pearson']:+.3f}$, RMSE ${CAL['rmse']:.3f}$")
ax[0].plot(g, np.clip(ap + bp * g, 0.4, 1.0), "--", color="tab:green", lw=2.2,
           label=rf"push PPR: ${ap:.3f}+{bp:.3f}\bar{{P}}_{{esc}}$"
                 "\n   $r=+0.961$, RMSE $0.068$")
ax[0].set_xlabel(r"mean escape probability $\bar{P}_{\mathrm{esc}}$")
ax[0].set_ylabel(r"empirically optimal $\alpha$")
ax[0].set_title("(a) Calibration on 54 training instances", fontsize=10)
ax[0].legend(fontsize=7, loc="lower right")
ax[0].set_ylim(0.42, 1.07)

# ---------------- B: out-of-sample alpha sweep ------------------------------
gr = np.array(SW["grid"])
sty = {"escape-scaled pull": ("tab:orange", "-"), "push PPR": ("tab:green", "--")}
for fam, (c, ls) in sty.items():
    d = SW[fam]
    ax[1].plot(gr, d["curve"], ls, color=c, lw=1.9, label=f"{fam}: fixed $\\alpha$")
    ax[1].axhline(d["rule_rho"], color=c, lw=1.1, alpha=0.55, ls=":")
    ax[1].plot([d["best_fixed_alpha"]], [d["best_fixed"]], "v", color=c, ms=7,
               markeredgecolor="k", markeredgewidth=0.4)
    ax[1].annotate(f"{fam}, rule-$\\alpha$: {d['rule_rho']:.3f}",
                   xy=(0.995, d["rule_rho"]), fontsize=7, color=c,
                   va="bottom", ha="right")
ax[1].set_xlabel(r"damping factor $\alpha$ (single value applied to all instances)")
ax[1].set_ylabel(r"mean Spearman $\rho$")
ax[1].set_title("(b) No fixed $\\alpha$ matches the per-instance rule", fontsize=10)
ax[1].legend(fontsize=7, loc="lower right", framealpha=0.93)
ax[1].set_ylim(0.32, 0.99)

# ---------------- C: baselines ---------------------------------------------
recs = HO["records"]
push_rule = float(np.mean(AT["results"]["PPR push|rule"]))
show = [("x only", "base", "$x$ only"),
        ("PPR pull rule-a", "base", "PPR pull"),
        ("P_esc alone", "base", r"$P_{\mathrm{esc}}$"),
        ("sum_j p_ij", "base", r"$\sum_j p_{ij}$"),
        ("MC 10 samples", "samp", "MC 10"),
        ("PPR push a=.85", "base", r"push PPR $\alpha{=}.85$"),
        ("MIA/PMIA (KDD10)", "base", "MIA/PMIA"),
        ("MC 50 samples", "samp", "MC 50"),
        ("THIS WORK", "ours", "escape-scaled rule-$\\alpha$"),
        ("__push_rule__", "ours", "push PPR rule-$\\alpha$"),
        ("oracle a", "orac", "oracle $\\alpha$"),
        ("IRIE (ICDM12)", "base", "IRIE"),
        ("MC 200 samples", "samp", "MC 200")]
vals = [push_rule if k == "__push_rule__"
        else float(np.mean([r["rho"][k] for r in recs])) for k, _, _ in show]
order = np.argsort(vals)
cols = {"base": "0.74", "samp": "tab:blue", "ours": "tab:orange", "orac": "0.40"}
for pos, i in enumerate(order):
    ax[2].bar(pos, vals[i], color=cols[show[i][1]], edgecolor="k", linewidth=0.5)
    ax[2].text(pos, vals[i] + 0.014, f"{vals[i]:.3f}", ha="center", fontsize=6.3)
ax[2].set_xticks(range(len(show)))
ax[2].set_xticklabels([show[i][2] for i in order],
                      fontsize=6.3, rotation=45, ha="right")
ax[2].set_ylabel(r"mean Spearman $\rho$")
ax[2].set_ylim(0, 1.06)
ax[2].set_title(r"(c) Out-of-sample comparison, $n=100$, 100 instances", fontsize=10)
h = [plt.Rectangle((0, 0), 1, 1, fc=cols[c], ec="k", lw=0.5)
     for c in ("base", "samp", "ours", "orac")]
ax[2].legend(h, ["heuristic baseline", "sampling estimator",
                 "calibrated by our rule", "per-instance oracle"],
             fontsize=6.4, loc="upper left", framealpha=0.93)

# ---------------- D: cost/accuracy -----------------------------------------
mc = SI["mc_curve"]
bs = sorted(int(b) for b in mc)
ax[3].plot([mc[str(b)]["ms"] for b in bs], [mc[str(b)]["rho"] for b in bs],
           "o-", color="tab:blue", ms=4, lw=1.5, label="Monte Carlo (5-320 samples)")
for b in bs:
    if b in (5, 20, 80, 320):
        ax[3].annotate(f"{b}", (mc[str(b)]["ms"], mc[str(b)]["rho"]),
                       textcoords="offset points", xytext=(3, -9),
                       fontsize=6, color="tab:blue")
o = SI["ours_small"]
ax[3].plot([o["ms"]], [o["rho"]], "*", color="tab:orange", ms=17,
           markeredgecolor="k", markeredgewidth=0.5, zorder=5,
           label=f"escape-scaled rule-$\\alpha$ ({o['rho']:.3f})")
n100 = {k: float(np.mean([r["rho"][k] for r in recs]))
        for k in ("IRIE (ICDM12)", "MIA/PMIA (KDD10)", "P_esc alone")}
t100 = {k: float(np.mean([r["ms"][k] for r in recs])) for k in n100}
for k, mk, c, lb in [("IRIE (ICDM12)", "s", "tab:green", "IRIE"),
                     ("MIA/PMIA (KDD10)", "^", "tab:red", "MIA/PMIA"),
                     ("P_esc alone", "D", "0.4", r"$P_{\mathrm{esc}}$ alone")]:
    ax[3].plot([t100[k]], [n100[k]], mk, color=c, ms=6.5,
               markeredgecolor="k", markeredgewidth=0.4, label=lb)
ax[3].set_xscale("log")
ax[3].set_xlabel("wall-clock per instance (ms, log scale)")
ax[3].set_ylabel(r"mean Spearman $\rho$")
ax[3].set_title("(d) Cost / accuracy", fontsize=10)
ax[3].legend(fontsize=6.6, loc="lower right", framealpha=0.93)

plt.tight_layout()
plt.savefig("fig1.pdf", bbox_inches="tight")
plt.savefig("fig1.png", bbox_inches="tight", dpi=150)
print("wrote fig1.pdf / fig1.png")