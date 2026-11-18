"""BRIGHT-Pro alpha-nDCG@10 (alpha = 0.5) and aspect recall A-Recall@10, using the evaluation functions of the
official BRIGHT-Pro repository (https://github.com/yale-nlp/Bright-Pro). Queries without gold documents are skipped;
the reported numbers are macro averages over the 7 domains.

Usage:
  git clone https://github.com/yale-nlp/Bright-Pro.git external/Bright-Pro
  python eval/eval_bright_pro.py --results results/coex --bright_pro_repo external/Bright-Pro
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

DOMAINS = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living"]


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/coex", help="directory with bp_<domain>/ranking.json")
    ap.add_argument("--bright_pro_repo", default="external/Bright-Pro")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    sys.path.insert(0, a.bright_pro_repo)
    from bright_pro_data import build_aspect_weights, build_doc_to_aspect_id, load_bright_pro
    A = _module(os.path.join(a.bright_pro_repo, "retrieval", "evaluation", "alpha-ndcg-evaluation.py"), "alpha_ndcg")
    R = _module(os.path.join(a.bright_pro_repo, "retrieval", "evaluation", "weighted_aspect_recall.py"), "aspect_recall")
    al, ar = [], []
    for t in DOMAINS:
        rk = json.load(open(os.path.join(a.results, f"bp_{t}", "ranking.json")))
        d2a, aw = build_doc_to_aspect_id(t), build_aspect_weights(t)
        qs = [(str(e["id"]), [g for g in e["gold_ids"] if g and g != "N/A"]) for e in load_bright_pro("examples", t)]
        qs = [(qid, g) for qid, g in qs if g]
        x = np.mean([100 * A.compute_alpha_dcg_at_k(rk[qid], g, d2a, aw, 0.5, a.k)
                     / max(1e-9, A.compute_alpha_idcg_at_k(g, d2a, aw, 0.5, a.k)) for qid, g in qs])
        y = np.mean([100 * R.compute_weighted_aspect_recall_at_k(rk[qid], g, d2a, aw, a.k) for qid, g in qs])
        al.append(x); ar.append(y)
        print(f"bp_{t:20s} alpha-nDCG@{a.k} {x:.1f}  A-Recall@{a.k} {y:.1f}  ({len(qs)} queries)")
    print(f"{'average':23s} alpha-nDCG@{a.k} {np.mean(al):.1f}  A-Recall@{a.k} {np.mean(ar):.1f}")


if __name__ == "__main__":
    main()
