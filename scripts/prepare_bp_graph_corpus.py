"""Write the BRIGHT-Pro corpora in the .jsonl format expected by GAREN's graph builder
(bash/build_group/build_graph.sh), so that the kNN graphs of the BRIGHT-Pro domains can be built with the
same procedure (Qwen3-Embedding-4B, exact cosine, k=16) as the released BRIGHT graphs.

Usage: python scripts/prepare_bp_graph_corpus.py <GAREN repo>/data/corpus
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from coex.data import load_task  # noqa: E402

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
for t in ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living"]:
    docs, _ = load_task(f"bp_{t}")
    with open(os.path.join(out_dir, f"bp_{t}.jsonl"), "w") as f:
        for d in docs:
            f.write(json.dumps({"docno": d["id"], "content": d["content"], "title": "", "paragraph_text": d["content"]}) + "\n")
    print(f"bp_{t}: {len(docs)} documents")
