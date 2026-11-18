"""Load BRIGHT (xlangai/BRIGHT) and BRIGHT-Pro (yale-nlp/Bright-Pro) from the Hugging Face Hub.

A task name with the prefix ``bp_`` (e.g. ``bp_biology``) refers to the BRIGHT-Pro subset of that domain.
Returns documents [{"id", "content"}] and queries [{"id", "query", "gold", "excluded"}].

Dataset revisions are pinned to the versions used in the paper. Note that xlangai/BRIGHT was updated on
2026-11-07 (relevance labels of biology, earth_science, robotics and stackoverflow corrected); all results in
the paper use the earlier revision below.
"""
from datasets import load_dataset

BRIGHT_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
BRIGHT_PRO_REVISION = "dbdc22babbef310210e267b99249e7cec86d5edf"


def _clean(xs):
    return [x for x in list(xs or []) if x and x != "N/A"]


def load_task(task):
    if task.startswith("bp_"):
        t = task[3:]
        d = load_dataset("yale-nlp/Bright-Pro", "documents", split=t, revision=BRIGHT_PRO_REVISION)
        e = load_dataset("yale-nlp/Bright-Pro", "examples", split=t, revision=BRIGHT_PRO_REVISION)
        docs = [{"id": r["id"], "content": r["content"]} for r in d]
        queries = [{"id": str(r["id"]), "query": r["query"], "gold": _clean(r["gold_ids"]), "excluded": set()} for r in e]
        return docs, queries
    d = load_dataset("xlangai/BRIGHT", "documents", split=task, revision=BRIGHT_REVISION)
    e = load_dataset("xlangai/BRIGHT", "examples", split=task, revision=BRIGHT_REVISION)
    docs = [{"id": r["id"], "content": r["content"]} for r in d]
    queries = [{"id": str(r["id"]), "query": r["query"], "gold": _clean(r["gold_ids"]),
                "excluded": set(_clean(r["excluded_ids"]))} for r in e]
    return docs, queries
