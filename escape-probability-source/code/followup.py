"""
Follow-up study at n=900 -- stratified random design, resumable.

Usage:
    python3 followup.py sample          # draw and save 400 configs
    python3 followup.py run <count>     # evaluate up to <count> pending configs

Results are appended to followup_n900.jsonl, one record per line, so the job
can be resumed after an interruption without losing work.
"""

import json
import os
import sys
import time
import numpy as np
from escape_lib import (make_instance, ground_truth, rho, timed, diffuse,
                        alpha_from_rule, m_pesc, m_sum_p, m_irie, m_mia,
                        m_mc, m_escape_pull)

CFG_FILE = "followup_configs.json"
OUT_FILE = "followup_n900.jsonl"

N = 900
MC_GT = 3000
N_BINS = 8
PER_BIN = 50
TOPOS = ["er", "ba", "grid", "ws"]
DEGS = [4, 6, 8, 10, 12]
PRIORS = ["dirichlet", "uniform", "sparse"]
ALPHA_GRID = np.round(np.arange(0.40, 1.001, 0.025), 4)

CAL = json.load(open("calibration.json"))
COEF = (CAL["intercept"], CAL["slope"])
AT = json.load(open("alpha_transfer.json"))
PUSH_COEF = tuple(AT["rules"]["PPR push"])


def push_rule(inst):
    a = float(np.clip(PUSH_COEF[0] + PUSH_COEF[1] * inst.mean_pesc, 0.4, 1.0))
    return diffuse(inst, a, use_pesc=False, orientation="push")


METHODS = {
    "P_esc alone":     lambda i: m_pesc(i),
    "sum_j p_ij":      lambda i: m_sum_p(i),
    "MIA/PMIA":        lambda i: m_mia(i, 1e-3),
    "IRIE":            lambda i: m_irie(i, 0.7),
    "push PPR rule-a": push_rule,
    "escape rule-a":   lambda i: m_escape_pull(i, alpha_from_rule(i, COEF)),
    "MC 50":           lambda i: m_mc(i, 50),
    "MC 200":          lambda i: m_mc(i, 200),
}


def do_sample():
    """Stratified-random draw: 8 equal-width Pesc strata, 50 instances each.

    Every non-stratifying factor (topology, average degree, probability range,
    prior type, seed) is drawn at random.  Average degree and prior type were
    held FIXED in the original 36-instance grid; varying them here widens the
    population the conclusions apply to.
    """
    rng = np.random.default_rng(20260905)
    bins = [[] for _ in range(N_BINS)]
    attempts = 0
    t0 = time.time()
    while sum(len(b) for b in bins) < N_BINS * PER_BIN and attempts < 60000:
        attempts += 1
        topo = TOPOS[rng.integers(len(TOPOS))]
        deg = int(DEGS[rng.integers(len(DEGS))])
        p_lo = float(np.exp(rng.uniform(np.log(0.005), np.log(0.65))))
        p_hi = float(min(p_lo + rng.uniform(0.02, 0.30), 0.97))
        prior = PRIORS[rng.integers(len(PRIORS))]
        sd = int(rng.integers(1, 10**6))
        inst = make_instance(topo, N, p_lo, p_hi, sd, prior_kind=prior,
                             avg_deg=deg)
        b = min(int(inst.mean_pesc * N_BINS), N_BINS - 1)
        if len(bins[b]) >= PER_BIN:
            continue
        bins[b].append(dict(id=sum(len(z) for z in bins), topology=topo,
                            avg_deg=deg, p_lo=p_lo, p_hi=p_hi, prior=prior,
                            seed=sd, mean_pesc=float(inst.mean_pesc),
                            m=int(inst.m), bin=b))
    cfgs = [c for b in bins for c in b]
    for i, c in enumerate(cfgs):
        c["id"] = i
    json.dump(dict(configs=cfgs, attempts=attempts, n_bins=N_BINS,
                   per_bin=PER_BIN, n=N,
                   alpha_grid=[float(a) for a in ALPHA_GRID],
                   coef_escape=list(COEF), coef_push=list(PUSH_COEF)),
              open(CFG_FILE, "w"))
    print(f"sampled {len(cfgs)} configs in {attempts} attempts "
          f"({time.time()-t0:.1f}s)")
    for k, b in enumerate(bins):
        pes = [c["mean_pesc"] for c in b]
        print(f"  bin {k} Pesc [{k/N_BINS:.3f},{(k+1)/N_BINS:.3f}): "
              f"{len(b):3d}  mean {np.mean(pes):.3f}")


def evaluate(cfg):
    inst = make_instance(cfg["topology"], N, cfg["p_lo"], cfg["p_hi"],
                         cfg["seed"], prior_kind=cfg["prior"],
                         avg_deg=cfg["avg_deg"])
    gt = ground_truth(inst, MC_GT, seed=cfg["seed"] % 10**6 + 1)
    rec = {k: cfg[k] for k in ("id", "topology", "avg_deg", "p_lo", "p_hi",
                               "prior", "seed", "mean_pesc", "m", "bin")}
    rec["rho"] = {}
    rec["ms"] = {}
    for nm, fn in METHODS.items():
        s, dt = timed(fn, inst)
        rec["rho"][nm] = rho(s, gt)
        rec["ms"][nm] = dt * 1000
    ce = np.array([rho(diffuse(inst, float(a), use_pesc=True,
                               orientation="pull"), gt) for a in ALPHA_GRID])
    cp = np.array([rho(diffuse(inst, float(a), use_pesc=False,
                               orientation="push"), gt) for a in ALPHA_GRID])
    rec["oracle_escape"] = float(ce.max())
    rec["alpha_opt_escape"] = float(ALPHA_GRID[int(ce.argmax())])
    rec["oracle_push"] = float(cp.max())
    rec["alpha_opt_push"] = float(ALPHA_GRID[int(cp.argmax())])
    rec["curve_escape"] = [round(float(v), 6) for v in ce]
    rec["curve_push"] = [round(float(v), 6) for v in cp]
    return rec


def do_run(count):
    cfgs = json.load(open(CFG_FILE))["configs"]
    done = set()
    if os.path.exists(OUT_FILE):
        for line in open(OUT_FILE):
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    pending = [c for c in cfgs if c["id"] not in done]
    todo = pending[:count]
    print(f"{len(done)} done, {len(pending)} pending; evaluating {len(todo)}")
    t0 = time.time()
    with open(OUT_FILE, "a") as fh:
        for k, cfg in enumerate(todo):
            rec = evaluate(cfg)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (k + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  {k+1}/{len(todo)}  {el/60:.1f} min  "
                      f"({el/(k+1):.2f} s/inst)", flush=True)
    print(f"chunk done in {(time.time()-t0)/60:.1f} min; "
          f"total complete {len(done)+len(todo)}/{len(cfgs)}")


if __name__ == "__main__":
    if sys.argv[1] == "sample":
        do_sample()
    elif sys.argv[1] == "run":
        do_run(int(sys.argv[2]))
