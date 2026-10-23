"""Figure 4: real-world validation."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REAL = [json.loads(l) for l in open("real_results.jsonl") if l.strip()]
DENSE = [json.loads(l) for l in open("dense_panel.jsonl")
         if l.strip() and json.loads(l)["n"] > 100]
SCALE = [json.loads(l) for l in open("scale_real.jsonl") if l.strip()]

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 160})
fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
ax = axes.ravel()

# ---- (a) citation networks: the router's buckets vs reality ---------------
E = [0, .625, .875, 1.01]
NAMES = ["$<0.625$\n(routed: IRIE)", "$0.625$–$0.875$\n(routed: MC 200)",
         "$\\geq 0.875$\n(routed: $P_{esc}$)"]
M = ["P_esc alone", "IRIE", "escape rule-a", "MC 200"]
COL = {"P_esc alone": "0.35", "IRIE": "tab:green",
       "escape rule-a": "tab:orange", "MC 200": "tab:blue"}
LB = {"P_esc alone": r"$P_{esc}$ alone", "IRIE": "IRIE",
      "escape rule-a": "escape rule-$\\alpha$", "MC 200": "MC 200"}
w = 0.2
for k, m in enumerate(M):
    ys = []
    for b in range(3):
        sel = [r for r in REAL if E[b] <= r["mean_pesc"] < E[b + 1]]
        ys.append(np.mean([r["rho"][m] for r in sel]) if sel else 0)
    ax[0].bar(np.arange(3) + (k - 1.5) * w, ys, w, color=COL[m],
              edgecolor="k", lw=.4, label=LB[m])
ax[0].set_xticks(range(3))
ax[0].set_xticklabels(NAMES, fontsize=7)
ax[0].set_ylabel(r"mean Spearman $\rho$")
ax[0].set_ylim(0, 1.08)
ax[0].set_title("(a) Citation networks: sampling wins every bucket",
                fontsize=10)
ax[0].legend(fontsize=6.6, ncol=2, loc="lower left", framealpha=.93)

# ---- (b) dense regime, by genre -------------------------------------------
genres = ["amazon-photo", "amazon-computers", "coauthor-cs",
          "coauthor-physics"]
pe = [np.mean([r["rho"]["P_esc alone"] for r in DENSE if r["genre"] == g])
      for g in genres]
mc = [np.mean([r["rho"]["MC 200"] for r in DENSE if r["genre"] == g])
      for g in genres]
mc8 = [np.mean([r["rho"]["MC 800"] for r in DENSE if r["genre"] == g])
       for g in genres]
xp = np.arange(len(genres))
ax[1].bar(xp - .25, pe, .25, color="0.35", edgecolor="k", lw=.4,
          label=r"$P_{esc}$ alone (free)")
ax[1].bar(xp, mc, .25, color="tab:blue", edgecolor="k", lw=.4,
          label="MC 200 (263 ms)")
ax[1].bar(xp + .25, mc8, .25, color="tab:cyan", edgecolor="k", lw=.4,
          label="MC 800 (1035 ms)")
ax[1].set_xticks(xp)
ax[1].set_xticklabels([g.replace("-", "\n") for g in genres], fontsize=6.8)
ax[1].set_ylim(0.93, 1.005)
ax[1].set_ylabel(r"mean Spearman $\rho$")
ax[1].set_title(r"(b) Dense regime ($\bar{P}_{esc}=0.90$), 30 instances",
                fontsize=10)
ax[1].legend(fontsize=6.8, loc="lower left", framealpha=.93)

# ---- (c) paired differences in the dense regime ---------------------------
d200 = np.array([r["rho"]["P_esc alone"] - r["rho"]["MC 200"] for r in DENSE])
d800 = np.array([r["rho"]["P_esc alone"] - r["rho"]["MC 800"] for r in DENSE])
d50 = np.array([r["rho"]["P_esc alone"] - r["rho"]["MC 50"] for r in DENSE])
bp = ax[2].boxplot([d50, d200, d800], vert=False, widths=.6,
                   showfliers=False, patch_artist=True,
                   medianprops=dict(color="k", lw=1.4))
for patch, c in zip(bp["boxes"], ["tab:cyan", "tab:blue", "navy"]):
    patch.set_facecolor(c)
    patch.set_alpha(.55)
for i, d in enumerate([d50, d200, d800]):
    ax[2].plot([d.mean()], [i + 1], "D", color="k", ms=5, zorder=5)
ax[2].axvline(0, color="k", lw=1.2, ls="--")
ax[2].set_yticklabels(["vs MC 50\n$p=4\\times10^{-9}$",
                       "vs MC 200\n$p=4\\times10^{-4}$",
                       "vs MC 800\n$p=0.32$"], fontsize=7)
ax[2].set_xlabel(r"paired difference in $\rho$ ($P_{esc}$ alone minus sampling)")
ax[2].set_title("(c) The free score beats 200-sample MC", fontsize=10)

# ---- (d) scale sweep, regime held fixed -----------------------------------
for tgt, c, mk in [(0.6, "tab:green", "o"), (0.9, "tab:red", "s")]:
    sel = sorted([r for r in SCALE if r["target"] == tgt],
                 key=lambda z: z["n"])
    if not sel:
        continue
    ns = [r["n"] for r in sel]
    ax[3].plot(ns, [r["rho"]["MC 200"] for r in sel], "-" + mk, color=c,
               ms=5, lw=1.6, label=f"MC 200, $\\bar{{P}}_{{esc}}={tgt}$")
    ax[3].plot(ns, [r["rho"]["P_esc alone"] for r in sel], "--" + mk,
               color=c, ms=5, lw=1.6, alpha=.6, mfc="none",
               label=f"$P_{{esc}}$ alone, $\\bar{{P}}_{{esc}}={tgt}$")
ax[3].set_xscale("log")
ax[3].set_xlabel("$n$ (log scale), coauthor-physics subgraphs")
ax[3].set_ylabel(r"Spearman $\rho$")
ax[3].set_title("(d) Scale, with the density regime held fixed", fontsize=10)
ax[3].legend(fontsize=6.4, loc="lower left", framealpha=.93)

plt.tight_layout()
plt.savefig("../fig4.pdf", bbox_inches="tight")
plt.savefig("../fig4.png", bbox_inches="tight", dpi=150)
print("wrote fig4.pdf / fig4.png")
