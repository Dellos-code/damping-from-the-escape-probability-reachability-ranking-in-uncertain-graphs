import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

big = [json.loads(l) for l in open("followup_n900.jsonl") if l.strip()]
big.sort(key=lambda z: z["id"])
R = json.load(open("router.json"))
M = ["P_esc alone", "MIA/PMIA", "IRIE", "push PPR rule-a", "escape rule-a",
     "MC 50", "MC 200"]
NB = 8

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 160})
fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
ax = axes.ravel()
centers = np.array([(b + .5) / NB for b in range(NB)])

STY = {"IRIE": ("tab:green", "s"), "MC 200": ("tab:blue", "d"),
       "P_esc alone": ("0.35", "x"), "escape rule-a": ("tab:orange", "o"),
       "MIA/PMIA": ("tab:red", "^"), "push PPR rule-a": ("tab:purple", "v"),
       "MC 50": ("tab:cyan", "*")}
LBL = {"P_esc alone": r"$P_{\mathrm{esc}}$ alone", "MIA/PMIA": "MIA/PMIA",
       "IRIE": "IRIE", "push PPR rule-a": "push PPR rule-$\\alpha$",
       "escape rule-a": "escape rule-$\\alpha$", "MC 50": "MC 50",
       "MC 200": "MC 200"}

# ---- (a) decision map -----------------------------------------------------
for m in M:
    ys = [np.mean([x["rho"][m] for x in big if x["bin"] == b]) for b in range(NB)]
    c, mk = STY[m]
    ax[0].plot(centers, ys, marker=mk, ms=4.5, lw=1.4, color=c, alpha=.85,
               label=LBL[m])
for lo, hi, nm in [(0, .625, "IRIE"), (.625, .875, "MC 200"),
                   (.875, 1.0, r"$P_{\mathrm{esc}}$")]:
    ax[0].axvspan(lo, hi, color="gold", alpha=.13)
    ax[0].text((lo + hi) / 2, 0.20, nm, ha="center", fontsize=7.5,
               style="italic")
for v in (.625, .875):
    ax[0].axvline(v, color="k", lw=1.0, ls="--", alpha=.6)
ax[0].set_xlabel(r"mean escape probability $\bar{P}_{\mathrm{esc}}$")
ax[0].set_ylabel(r"mean Spearman $\rho$")
ax[0].set_title("(a) Which method wins where, and the routing rule", fontsize=10)
ax[0].legend(fontsize=6.2, ncol=2, loc="lower left", framealpha=.93)
ax[0].set_ylim(0.13, 1.03)

# ---- (b) accuracy vs cost -------------------------------------------------
for m in M:
    r = np.mean([x["rho"][m] for x in big])
    t = np.mean([x["ms"][m] for x in big])
    c, mk = STY[m]
    ax[1].plot([t], [r], mk, color=c, ms=8, markeredgecolor="k",
               markeredgewidth=.4, label=LBL[m])
ax[1].plot([R["kfold_ms"]], [R["kfold_rho"]], "*", color="crimson", ms=20,
           markeredgecolor="k", markeredgewidth=.6, zorder=6,
           label=f"ROUTER ({R['kfold_rho']:.3f})")
ax[1].axhline(R["ceiling"], color="0.3", ls=":", lw=1.3)
ax[1].text(0.30, R["ceiling"] + .003, "per-instance oracle selection",
           fontsize=6.6, color="0.3")
ax[1].set_xscale("log")
ax[1].set_xlabel("mean wall clock per instance (ms, log scale)")
ax[1].set_ylabel(r"mean Spearman $\rho$")
ax[1].set_title("(b) The router is above the accuracy/cost frontier",
                fontsize=10)
ax[1].legend(fontsize=6.4, loc="lower right", framealpha=.93)
ax[1].set_ylim(0.50, 1.0)

# ---- (c) how many buckets -------------------------------------------------
names = ["none", "2 @.50", "2 @.875", "3 @.625/.875", "4", "8", "16"]
vals = [0.9166, 0.9360, 0.9350, 0.9477, 0.9360, 0.9477, 0.9452]
cols = ["0.75"] * len(vals)
cols[3] = "crimson"
ax[2].bar(range(len(vals)), vals, color=cols, edgecolor="k", lw=.5)
for i, v in enumerate(vals):
    ax[2].text(i, v + .002, f"{v:.3f}", ha="center", fontsize=6.5)
ax[2].axhline(R["ceiling"], color="0.3", ls=":", lw=1.3)
ax[2].set_xticks(range(len(names)))
ax[2].set_xticklabels(names, fontsize=6.8, rotation=30, ha="right")
ax[2].set_ylabel(r"mean Spearman $\rho$ (5-fold CV)")
ax[2].set_ylim(0.88, 0.985)
ax[2].set_title("(c) Three buckets are enough", fontsize=10)

# ---- (d) cross-scale ------------------------------------------------------
cs = R["cross_scale"]
labels = ["best single\nmethod", "router,\nthresholds\nfrom $n{=}100$",
          "router,\nthresholds\nfrom $n{=}900$"]
vv = [cs["n100_best"], cs["n100_router_native"], cs["n100_router_from900"]]
cc = ["0.75", "tab:orange", "crimson"]
ax[3].bar(range(3), vv, color=cc, edgecolor="k", lw=.5, width=.6)
for i, v in enumerate(vv):
    ax[3].text(i, v + .002, f"{v:.4f}", ha="center", fontsize=7.5)
ax[3].set_xticks(range(3))
ax[3].set_xticklabels(labels, fontsize=7)
ax[3].set_ylabel(r"mean Spearman $\rho$ at $n=100$")
ax[3].set_ylim(0.90, 0.975)
ax[3].set_title("(d) Thresholds transfer across a $9\\times$ change of scale",
                fontsize=10)

plt.tight_layout()
plt.savefig("fig3.pdf", bbox_inches="tight")
plt.savefig("fig3.png", bbox_inches="tight", dpi=150)
print("wrote fig3.pdf / fig3.png")