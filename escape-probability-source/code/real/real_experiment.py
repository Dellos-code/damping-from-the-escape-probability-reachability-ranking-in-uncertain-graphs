"""
Real-world validation.

Everything here is applied WITHOUT REFITTING:
  * the 3-bucket router with thresholds fitted at n=900 on synthetic graphs
        Pesc_bar < 0.625        -> IRIE
        0.625 <= Pesc_bar < .875-> Monte Carlo, 200 samples
        Pesc_bar >= 0.875       -> Pesc alone
  * the damping rules alpha = clip(a + b*Pesc_bar, .4, 1) fitted at n=64

Topologies: Cora, CiteSeer and PubMed citation networks (Planetoid release) plus
their k-cores, which give real structure at a range of densities.  These have
genuine degree heterogeneity (CV 0.8-1.3, hubs 30-40x the mean) and real
clustering (0.14-0.38), neither of which any of the four synthetic families
reproduces.

Edge-probability models:
  uniform ranges (as in training) at four levels, plus WEIGHTED CASCADE
  p_ij = 1/max(deg_i, deg_j) -- the standard IC assignment, and genuinely
  out of distribution for our rules.

Usage:  python3 real_experiment.py build      # cache the topologies
        python3 real_experiment.py run <k>    # evaluate k pending instances
Results append to real_results.jsonl (resumable).
"""

import json
import os
import pickle
import sys
import time
import numpy as np
import networkx as nx

sys.path.insert(0, "..")
from escape_lib import (Instance, ground_truth, rho, timed, diffuse,
                        alpha_from_rule, m_pesc, m_sum_p, m_irie, m_mia,
                        m_mc, m_escape_pull)

CAL = json.load(open("../calibration.json"))
COEF = (CAL["intercept"], CAL["slope"])
AT = json.load(open("../alpha_transfer.json"))
PUSH_COEF = tuple(AT["rules"]["PPR push"])
ROUTER = [(0.0, 0.625, "IRIE"), (0.625, 0.875, "MC 200"),
          (0.875, 1.01, "P_esc alone")]

TOPO_FILE = "topologies.pkl"
CFG_FILE = "real_configs.json"
OUT = "real_results.jsonl"

PROB_MODELS = [
    ("uniform[0.02,0.08]", ("u", 0.02, 0.08)),
    ("uniform[0.10,0.25]", ("u", 0.10, 0.25)),
    ("uniform[0.25,0.45]", ("u", 0.25, 0.45)),
    ("uniform[0.45,0.75]", ("u", 0.45, 0.75)),
    ("weighted-cascade 1/deg", ("wc", 0, 0)),
]
PRIORS = ["dirichlet", "uniform", "sparse"]


def load_planetoid(name):
    g = pickle.load(open(f"ind.{name}.graph", "rb"), encoding="latin1")
    G = nx.Graph()
    for u, nb in g.items():
        for v in nb:
            if u != v:
                G.add_edge(int(u), int(v))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def build():
    specs = [("cora", 1), ("cora", 2), ("cora", 3),
             ("citeseer", 1), ("citeseer", 2),
             ("pubmed", 4), ("pubmed", 5)]
    topos = {}
    for name, k in specs:
        G = load_planetoid(name)
        C = nx.k_core(G, k) if k > 1 else G
        C = nx.convert_node_labels_to_integers(C)
        lab = f"{name}-{k}core" if k > 1 else name
        deg = np.array([d for _, d in sorted(C.degree())], dtype=float)
        ei, ej = [], []
        for u, v in C.edges():
            a, b = (u, v) if u < v else (v, u)
            ei.append(a)
            ej.append(b)
        topos[lab] = dict(n=C.number_of_nodes(), ei=np.array(ei, dtype=np.int32),
                          ej=np.array(ej, dtype=np.int32), deg=deg,
                          clust=float(nx.average_clustering(C)),
                          deg_cv=float(deg.std() / deg.mean()))
        print(f"  {lab:16s} n={topos[lab]['n']:5d} m={len(ei):6d} "
              f"avg_deg={deg.mean():5.2f} cv={topos[lab]['deg_cv']:.2f} "
              f"clust={topos[lab]['clust']:.3f}")
    pickle.dump(topos, open(TOPO_FILE, "wb"))
    cfgs = []
    for lab in topos:
        for pi, (pname, _) in enumerate(PROB_MODELS):
            cfgs.append(dict(id=len(cfgs), topo=lab, prob=pname, pidx=pi,
                             prior=PRIORS[len(cfgs) % 3],
                             seed=1000 + len(cfgs)))
    json.dump(cfgs, open(CFG_FILE, "w"))
    print(f"\n{len(topos)} topologies, {len(cfgs)} instances queued")


def make_real_instance(topos, cfg):
    T = topos[cfg["topo"]]
    n, ei, ej, deg = T["n"], T["ei"], T["ej"], T["deg"]
    rng = np.random.default_rng(cfg["seed"])
    kind, lo, hi = PROB_MODELS[cfg["pidx"]][1]
    if kind == "u":
        p = rng.uniform(lo, hi, size=len(ei))
    else:                                    # weighted cascade, symmetrised
        p = 1.0 / np.maximum(deg[ei], deg[ej])
    p = np.clip(p, 1e-4, 0.999)
    if cfg["prior"] == "dirichlet":
        x = rng.dirichlet(np.full(n, 0.7))
    elif cfg["prior"] == "uniform":
        x = np.full(n, 1.0 / n)
    else:
        x = np.zeros(n)
        idx = rng.choice(n, size=max(1, n // 10), replace=False)
        x[idx] = rng.dirichlet(np.full(len(idx), 1.0))
    return Instance(n, ei, ej, p, x / x.sum())


def route(pe):
    for lo, hi, m in ROUTER:
        if lo <= pe < hi:
            return m
    return ROUTER[-1][2]


def run(count):
    topos = pickle.load(open(TOPO_FILE, "rb"))
    cfgs = json.load(open(CFG_FILE))
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [c for c in cfgs if c["id"] not in done][:count]
    print(f"{len(done)} done, {len(cfgs)-len(done)} pending; running {len(todo)}")
    t0 = time.time()
    with open(OUT, "a") as fh:
        for cfg in todo:
            ins = make_real_instance(topos, cfg)
            gt = ground_truth(ins, 3000, seed=cfg["seed"] + 7)
            a_e = alpha_from_rule(ins, COEF)
            a_p = float(np.clip(PUSH_COEF[0] + PUSH_COEF[1] * ins.mean_pesc,
                                0.4, 1.0))
            METH = {
                "P_esc alone":     lambda i: m_pesc(i),
                "sum_j p_ij":      lambda i: m_sum_p(i),
                "IRIE":            lambda i: m_irie(i, 0.7),
                "push PPR rule-a": lambda i: diffuse(i, a_p, use_pesc=False,
                                                     orientation="push"),
                "escape rule-a":   lambda i: m_escape_pull(i, a_e),
                "MC 50":           lambda i: m_mc(i, 50),
                "MC 200":          lambda i: m_mc(i, 200),
            }
            if ins.n <= 4500:
                METH["MIA/PMIA"] = lambda i: m_mia(i, 1e-3)
            rec = dict(cfg)
            rec.update(n=ins.n, m=int(ins.m), mean_pesc=ins.mean_pesc,
                       alpha_rule=a_e, rho={}, ms={})
            for nm, fn in METH.items():
                s, dt = timed(fn, ins)
                rec["rho"][nm] = rho(s, gt)
                rec["ms"][nm] = dt * 1000
            rec["routed_to"] = route(ins.mean_pesc)
            rec["rho"]["ROUTER"] = rec["rho"][rec["routed_to"]]
            rec["ms"]["ROUTER"] = rec["ms"][rec["routed_to"]]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  [{cfg['id']:2d}] {cfg['topo']:16s} {cfg['prob']:22s} "
                  f"Pesc={ins.mean_pesc:.3f} -> {rec['routed_to']:12s} "
                  f"router={rec['rho']['ROUTER']:.3f} "
                  f"irie={rec['rho']['IRIE']:.3f}", flush=True)
    print(f"chunk done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        run(int(sys.argv[2]))
