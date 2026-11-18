"""BRIGHT nDCG@10 (official protocol: excluded documents are removed from the ranking; pytrec_eval ndcg_cut_10).
Reports each domain, the StackExchange / Coding / Theorem-based averages and the macro average over 12 domains.

Usage: python eval/eval_bright.py --results results/coex
"""
import argparse
import json
import os
import sys

import numpy as np
import pytrec_eval

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from coex.data import load_task  # noqa: E402

GROUPS = [("StackExchange", ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living"]),
          ("Coding", ["leetcode", "pony"]),
          ("Theorem-based", ["aops", "theoremqa_questions", "theoremqa_theorems"])]


def ndcg10(task, ranking):
    _, queries = load_task(task)
    queries = [q for q in queries if q["gold"]]
    qrels = {q["id"]: {g: 1 for g in q["gold"]} for q in queries}
    run = {}
    for q in queries:
        lst = [d for d in ranking.get(q["id"], []) if d not in q["excluded"]][:1000]
        run[q["id"]] = {d: float(len(lst) - i) for i, d in enumerate(lst)} or {"__none__": 0.0}
    per = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"}).evaluate(run)
    return 100 * float(np.mean([per[q["id"]]["ndcg_cut_10"] for q in queries]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/coex", help="directory with <task>/ranking.json")
    a = ap.parse_args()
    scores = {}
    for _, tasks in GROUPS:
        for t in tasks:
            scores[t] = ndcg10(t, json.load(open(os.path.join(a.results, t, "ranking.json"))))
            print(f"{t:22s} {scores[t]:.1f}")
    for g, tasks in GROUPS:
        print(f"{g + ' avg':22s} {np.mean([scores[t] for t in tasks]):.1f}")
    print(f"{'All (12 domains)':22s} {np.mean(list(scores.values())):.1f}")


if __name__ == "__main__":
    main()
