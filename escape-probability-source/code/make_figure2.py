import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = json.load(open("followup_configs.json"))
recs = [json.loads(l) for l in open("followup_n900.jsonl") if l.strip()]
recs.sort(key=lambda r: r["id"])
GRID = np.array(CFG["alpha_grid"])
NB = CFG["n_bins"]

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 160})
fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
ax = axes.ravel()

pe = np.array([r["mean_pesc"] for r in recs])
bins = np.array([r["bin"] for r in recs])
centers = np.array([(b + 0.5) / NB for b in range(NB)])

# --------------- (a) stratified performance --------------------------------
SHOW = [("escape rule-a", "tab:orange", "-", "o", "escape-scaled rule-$\\alpha$"),
        ("IRIE", "tab:green", "--", "s", "IRIE"),
        ("MIA/PMIA", "tab:red", "-.", "^", "MIA/PMIA"),
        ("push PPR rule-a", "tab:purple", ":", "v", "push PPR rule-$\\alpha$"),
        ("MC 200", "tab:blue", "-", "d", "Monte Carlo, 200"),
        ("P_esc alone", "0.35", "--", "x", r"$P_{\mathrm{esc}}$ alone")]
for key, c, ls, mk, lb in SHOW:
    ys = [np.mean([r["rho"][key] for r in recs if r["bin"] == b])
          for b in range(NB)]
    ax[0].plot(centers, ys, ls, color=c, marker=mk, ms=4.5, lw=1.6, label=lb)
ax[0].set_xlabel(r"mean escape probability $\bar{P}_{\mathrm{esc}}$ (stratum centre)")
ax[0].set_ylabel(r"mean Spearman $\rho$")
ax[0].set_title("(a) Performance by density stratum (50 instances each)",
                fontsize=10)
ax[0].legend(fontsize=6.6, loc="lower left", ncol=2, framealpha=0.93)
ax[0].set_ylim(0.15, 1.03)

# --------------- (b) alpha* vs Pesc at n=900 -------------------------------
ao = np.array([r["alpha_opt_escape"] for r in recs])
ax[1].scatter(pe, ao, s=11, alpha=0.35, edgecolor="none", color="tab:orange",
              label=r"per-instance optimal $\alpha$, $n{=}900$")
g = np.linspace(0, 1, 200)
a64, b64 = CFG["coef_escape"]
b900, a900 = np.polyfit(pe, ao, 1)
ax[1].plot(g, np.clip(a64 + b64 * g, 0.4, 1.0), "r-", lw=2.2,
           label=rf"rule fitted at $n{{=}}64$: ${a64:.3f}+{b64:.3f}\bar{{P}}_{{esc}}$")
ax[1].plot(g, np.clip(a900 + b900 * g, 0.4, 1.0), "--", color="k", lw=1.8,
           label=rf"refit at $n{{=}}900$: ${a900:.3f}+{b900:.3f}\bar{{P}}_{{esc}}$")
med = [np.median(ao[bins == b]) for b in range(NB)]
ax[1].plot(centers, med, "o-", color="darkorange", ms=6, lw=2,
           markeredgecolor="k", markeredgewidth=0.5, label="per-stratum median")
ax[1].set_xlabel(r"mean escape probability $\bar{P}_{\mathrm{esc}}$")
ax[1].set_ylabel(r"empirically optimal $\alpha$")
ax[1].set_title(r"(b) The rule extrapolates from $n{=}64$ to $n{=}900$",
                fontsize=10)
ax[1].legend(fontsize=6.5, loc="lower right", framealpha=0.93)
ax[1].set_ylim(0.37, 1.06)

# --------------- (c) rule vs fixed vs oracle -------------------------------
lbl, rule_v, fix_v, orc_v = [], [], [], []
for fam, ck, ok, rk in [("escape-scaled", "curve_escape", "oracle_escape",
                         "escape rule-a"),
                        ("push PPR", "curve_push", "oracle_push",
                         "push PPR rule-a")]:
    mc = np.array([r[ck] for r in recs]).mean(axis=0)
    lbl.append(fam)
    fix_v.append(float(mc.max()))
    rule_v.append(float(np.mean([r["rho"][rk] for r in recs])))
    orc_v.append(float(np.mean([r[ok] for r in recs])))
xp = np.arange(2)
w = 0.26
ax[2].bar(xp - w, fix_v, w, color="0.75", edgecolor="k", lw=0.5,
          label=r"best fixed $\alpha$ (hindsight)")
ax[2].bar(xp, rule_v, w, color="tab:orange", edgecolor="k", lw=0.5,
          label=r"rule-$\alpha$ (fitted at $n{=}64$)")
ax[2].bar(xp + w, orc_v, w, color="0.40", edgecolor="k", lw=0.5,
          label="per-instance oracle")
for i in range(2):
    for off, v in ((-w, fix_v[i]), (0, rule_v[i]), (w, orc_v[i])):
        ax[2].text(i + off, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.6)
ax[2].set_xticks(xp)
ax[2].set_xticklabels(lbl, fontsize=8.5)
ax[2].set_ylabel(r"mean Spearman $\rho$")
ax[2].set_ylim(0, 1.05)
ax[2].set_title(r"(c) The rule beats any constant $\alpha$, $n{=}900$",
                fontsize=10)
ax[2].legend(fontsize=6.8, loc="lower right", framealpha=0.93)

# --------------- (d) paired differences ------------------------------------
ref = np.array([r["rho"]["escape rule-a"] for r in recs])
cmp = [("IRIE", "tab:green"), ("MC 200", "tab:blue"),
       ("MIA/PMIA", "tab:red"), ("push PPR rule-a", "tab:purple")]
data = [ref - np.array([r["rho"][k] for r in recs]) for k, _ in cmp]
bp = ax[3].boxplot(data, vert=False, widths=0.6, showfliers=False,
                   patch_artist=True, medianprops=dict(color="k", lw=1.4))
for patch, (_, c) in zip(bp["boxes"], cmp):
    patch.set_facecolor(c)
    patch.set_alpha(0.55)
ax[3].axvline(0, color="k", lw=1.2, ls="--")
for i, d in enumerate(data):
    ax[3].plot([d.mean()], [i + 1], "D", color="k", ms=5, zorder=5)
ax[3].set_yticklabels(["IRIE", "MC 200", "MIA/PMIA", "push PPR\nrule-$\\alpha$"],
                      fontsize=7.5)
ax[3].set_xlabel(r"paired difference in $\rho$  (escape rule-$\alpha$ minus baseline)")
ax[3].set_title("(d) Paired differences, 400 instances", fontsize=10)
ax[3].text(0.985, 0.05, "diamond = mean, line = median",
           transform=ax[3].transAxes, fontsize=6.5, ha="right",
           bbox=dict(fc="white", ec="0.7", lw=0.5, alpha=0.9))

plt.tight_layout()
plt.savefig("fig2.pdf", bbox_inches="tight")
plt.savefig("fig2.png", bbox_inches="tight", dpi=150)
print("wrote fig2.pdf / fig2.png")