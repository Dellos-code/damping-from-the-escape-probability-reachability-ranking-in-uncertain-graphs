# Damping from the Escape Probability

A free calibration rule for diffusion-based reachability ranking in uncertain
graphs.

Ioannis Konstantellos — Computer Engineering and Informatics Department,
University of Patras.

---

## The problem

Rank the vertices of a probabilistic graph by expected reachable mass,

    y*_i = E[ sum_{v in C(i)} x_v ]

where each edge (i,j) is independently present with probability p_ij and x is a
prior over which vertex is the search target. This equals single-seed
node-weighted influence spread under the Independent Cascade model, and equals
Edge Percolated Component centrality in the uniform case. Both identities are
verified numerically (notebook §2). Exact computation is #P-hard, so the field
uses fast heuristics — and a large family of those are damped diffusions whose
damping factor nobody chooses principledly.

## The result

The **mean escape probability**

    P̄esc = mean_i [ 1 - prod_j (1 - p_ij) ]

is computed while the transition operator is built, so it is free. It predicts
the best-performing damping factor:

    alpha = clip(0.502 + 0.558 * P̄esc, 0.4, 1.0)

Fitted on 54 instances at n=64 with Pearson r = +0.965, RMSE 0.043.

| claim | evidence |
|---|---|
| beats the conventional α = 0.85 | 0.894 vs 0.637 |
| beats the **best constant α selectable with hindsight** | 0.894 vs 0.677 |
| recovers a per-instance oracle | 98.2% |
| refits to a second operator family | push PPR: 0.241 + 0.859 P̄esc, r = +0.961, 98.7% of oracle |
| survives 14× change of scale, unrefitted | 95.5% of oracle at n=900 |
| tolerates rounding | `0.5 + 0.6 P̄esc` costs only 0.009 |
| beats every constant α on real networks | 91% / 89% of oracle |

Sensitivity is asymmetric: raising α is cheap (+0.10 on the intercept costs
0.031), lowering is expensive (−0.10 costs 0.199). When uncertain, err upward.

To our knowledge no prior rule sets α from an observable statistic available at
operator initialisation. The closest precedents — Kloumann–Ugander–Kleinberg
(PNAS 2017), Avrachenkov–Kadavankandy–Litvak (J. Stat. Phys. 2018), Li–Chien–
Milenkovic (NeurIPS 2019) — derive α inside planted models whose parameters
must be known; the paper positions against all three explicitly.

## Three secondary results, all bounding the first

**P_esc is not a new score.** It is exactly the correlation-free (η = 1) zeroth
iterate of Karrer–Newman–Zdeborová message-passing percolation (PRL 113:208702,
2014); the identity is verified numerically in notebook §14. So we implement
the full recursion as the proper deterministic competitor. On 30 real dense
instances:

| method | mean rho | ms |
|---|---|---|
| P_esc, i.e. k=0 | 0.9852 | ~0 |
| message passing, k=1 | 0.9919 | 6.0 |
| message passing, k=2 | 0.9928 | 3.2 |
| **message passing, converged** | **0.9931** | 38.0 |
| Monte Carlo, 200 samples | 0.9741 | 266.2 |

Message passing beats its own η=1 limit (+0.0079, p = 1.9e−9), so the paper
does not claim P_esc is the right deterministic score. The deterministic family
beats sampling (+0.0191, 29/30, p = 9.3e−9, at 1/7 the cost) — which confirms a
point Newman (Proc. R. Soc. A 479:20220774, 2023) already makes. What is ours:
**the ranking signal saturates after two iterations**, so the free iterate is
within one percent of convergence.

**The modern RR-set estimator trails plain Monte Carlo for whole-graph
ranking** (notebook §15; paper §7). Singleton-spread reverse-reachable sets
(Borgs'14 / IMM'15 lineage; lazy subset-sampled generation in the spirit of
SUBSIM, TODS 2022), on the same instances and ground truth:

| | synthetic (100) | citation (35) | dense real (30) |
|---|---|---|---|
| MC, 200 worlds | 0.934 / 28ms | 0.964 / 87ms | 0.974 / 265ms |
| RR, θ=200 | 0.801 / 31ms | 0.641 / 99ms | 0.972 / 556ms |
| RR, θ=800 | 0.924 / 123ms | 0.738 / 386ms | 0.989 / 2251ms |

At matched budget RR loses (synthetic −0.133, p=3.9e−16; citation −0.323,
p=2.9e−10, with a sparse-bucket collapse to 0.41). The cause is located: one
sampled world informs every vertex simultaneously; one RR set contributes one
containment bit per vertex, and in sparse regimes leaves most vertices tied at
zero hits. RR sets are the right primitive for top-k seed selection — their
design goal — not for whole-graph ranking. In the dense regime RR-800 finally
beats MC-200 (+0.015, 30/30) at 8.5× the cost, but not the free P_esc
(p=0.34), and converged message passing beats it outright at 1/59 the cost.

**Algorithm selection by P̄esc fails on real data.** A three-bucket router
beats the best single method on 400 synthetic instances (0.9477 vs 0.9166,
5-fold CV, p = 0.009) and collapses on real citation networks (0.9176 vs
0.9637 for just running Monte Carlo). Sampling wins *every* bucket there; a
refit router degenerates to selecting it everywhere. Reported as a negative
result.

## Contents

    paper.pdf / paper.tex     the paper (15 pp., 64 refs, pdflatex, no bibtex)
    fig1.pdf                  the calibration — the main result
    fig2.pdf                  stratified n=900 panel
    fig3.pdf                  the router (negative result)
    fig4.pdf                  real networks and message passing
    reproduce.ipynb           full reproduction, 68 cells
    RESPONSE_TO_EDITOR.md     point-by-point response to the KAIS desk reject
    code/                     scripts the notebook is assembled from
    code/rr_ranking.py        the RR-set estimator (lazy, frontier-vectorised)
    code/mp_percolation.py    message-passing percolation
    code/real/                real-network experiments
    data/, data/real/         result files regenerated/produced in this revision

`data/` holds the files this revision regenerated bit-identically
(calibration, hold-out, cross-operator transfer) plus the new RR panels;
the remaining panels (stratified n=900, router, diagnostics, per-node,
scale sweeps) regenerate from the notebook, checkpointed and resumable.
Every number in the paper is produced by a notebook section.

## Reproducing

    pip install numpy scipy networkx matplotlib
    jupyter notebook reproduce.ipynb

About 2.5 hours on one core. Long panels are checkpointed per instance and
resumable. Real network data is fetched from GitHub mirrors of the Planetoid
and gnn-benchmark releases; no other downloads are needed.

Reproduction has been verified across independent environments: the fitted
rule (0.502 + 0.558·P̄esc, r = +0.965, RMSE 0.043), the hold-out means
(0.894 / 0.904 / 98.2% / 98.7%), and the real-network means (MC-200 0.9637 on
citation; P_esc 0.9852 and MC-200 0.9741 on the dense panel) come out
bit-identical under NumPy's seeded default_rng.

## What this revision adds (post desk-reject)

- **A modern sampling baseline actually run**, not just cited: the RR-set
  estimator on all three panels, with the matched-budget result and its
  located cause (paper §7).
- **23 verified 2019–2026 references** spanning TKDE (×3), TKDD, KAIS, KDD,
  ICDE, SIGMOD (×3), TODS, PACMMOD, ICML, WSDM, ASONAM (×2), NeurIPS, ICLR,
  IJCAI, TCSS (×2), PNAS (×2), TSP, Neural Networks — every venue/year/page
  field checked against the publisher page before inclusion.
- **Explicit positioning against the closest precedents** for the α rule
  (Kloumann PNAS'17; Avrachenkov J.Stat.Phys.'18; Li–Chien–Milenkovic
  NeurIPS'19) and against learned alternatives (GPR-GNN, SuRe, Pirrò IJCAI'25).
- **A positioning paragraph among learned estimators** using only those
  papers' own reported costs (GLIE, DeepIM, ToupleGDD, PIANO) — no estimated
  numbers — including the observation that GLIE's first message-passing layer
  is built on exactly the escape probability.
- **A sharpened honesty clause**: on real networks the upper clip does much of
  the α rule's work; on synthetic panels a high fixed α is demonstrably not
  enough (0.677 vs 0.894).

## What is honest about this package

- **P_esc is not ours and is not the best deterministic score.** It is a known
  quantity in its most degenerate form. We use it as a free *feature*, not as
  a method.
- **The α rule is empirical.** No derivation. Its motivating correlation
  collapses on real networks (r = +0.965 → +0.225) even though the rule keeps
  working — and the paper says where the residual value comes from.
- **The router failure is reported, not buried.**
- **Edge probabilities are assigned, not observed.**
- **Modern baselines are scoped, not exhaustive**: RR is run in singleton-
  spread form; trained neural estimators are cited with their own reported
  costs and limits rather than run, and the paper says why.
- **Sample sizes and per-regime splits are stated everywhere.**

## Acknowledgement of method

The experimental design, ablations and statistical testing were carried out
with the assistance of an AI system. The escape-probability formulation and
the underlying model are the author's.
