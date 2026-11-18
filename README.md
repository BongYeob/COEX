# COEX

Code for the main experiments of COEX (Qwen3-Next-80B-A3B-Instruct-FP8 as the generator and listwise reranker) on
BRIGHT and BRIGHT-Pro.

COEX keeps several probes (competing explanations of the question, a shared-evidence probe and the original
query), each searching the whole corpus with BM25, and a judgment loop over a listwise LLM reranker
(window 20, 10 retained documents, 9 reranker calls). The loop accumulates judgments across rounds, re-judges
newly promoted documents before they are retained, allocates new-document slots among probes by their rewards,
adds kNN neighbours of strongly judged documents, and updates the probes' search texts with the knowledge that is
still missing. See the docstring of `coex/run.py` for the exact procedure.

## Layout

```
coex/run.py        COEX (probes, judgment loop, probe updates) and the entry point
coex/reranker.py   listwise reranker (reasoning-based prompt released with GAREN)
coex/bm25.py       BM25 identical to the official BRIGHT implementation
coex/data.py       BRIGHT / BRIGHT-Pro loaders (Hugging Face, pinned revisions)
eval/               nDCG@10 (BRIGHT), alpha-nDCG@10 and A-Recall@10 (BRIGHT-Pro)
scripts/            vLLM server, main run, BRIGHT-Pro corpus export for graph building
```

## Setup

```bash
pip install -r requirements.txt          # Python 3.11
# Pyserini needs Java 11: set JAVA_HOME (and JVM_PATH if needed)
git clone https://github.com/yale-nlp/Bright-Pro.git external/Bright-Pro   # official BRIGHT-Pro evaluation
```

**Data.** BRIGHT and BRIGHT-Pro are loaded from the Hugging Face Hub (`xlangai/BRIGHT`, `yale-nlp/Bright-Pro`).
The revisions are pinned in `coex/data.py`. `xlangai/BRIGHT` was updated on 2026-11-07 (corrected relevance labels
in biology, earth_science, robotics and stackoverflow); the paper uses the earlier revision `3066d29`.

**Document kNN graphs.** The kNN fill uses document graphs with 16 neighbours per document built from
Qwen3-Embedding-4B embeddings (exact cosine), in the format released by GAREN.
- BRIGHT: download `data/GRAPH/qwen3/graph/<task>/` from the GAREN data release
  ([`Jun421/GAREN-data`](https://huggingface.co/datasets/Jun421/GAREN-data)) into `data/GRAPH/qwen3/graph/`.
- BRIGHT-Pro: build the graphs with GAREN's graph builder (`bash/build_group/build_graph.sh` in
  https://github.com/KJ12124/GAREN), after exporting the corpora:
  ```bash
  python scripts/prepare_bp_graph_corpus.py <GAREN>/data/corpus
  cd <GAREN> && TASKS="bp_biology bp_earth_science bp_economics bp_psychology bp_robotics bp_stackoverflow bp_sustainable_living" \
      bash bash/build_group/build_graph.sh
  ```
  and copy `<GAREN>/data/GRAPH/qwen3/graph/bp_*` into `data/GRAPH/qwen3/graph/`.

## Run

```bash
bash scripts/serve_vllm.sh 0,1 8000            # Qwen3-Next-80B-A3B-Instruct-FP8, tensor parallel on 2 GPUs
bash scripts/run_main.sh http://localhost:8000/v1
```

`run_main.sh` runs the 12 BRIGHT domains, then the 7 BRIGHT-Pro domains, and evaluates. The probes (explanations and
shared evidence) are generated once per BRIGHT domain and cached in `cache/probes/`; each BRIGHT-Pro domain has the
same query ids as its BRIGHT domain and reuses that cache. A single domain:

```bash
python -m coex.run --task biology --url http://localhost:8000/v1
python -m coex.run --task bp_biology --url http://localhost:8000/v1
```

Outputs go to `results/coex/<task>/` (`ranking.json`: final ranking per query; `trace.json`: reranker windows,
probe rewards and probe texts per round; `config.json`).

Main settings (defaults in `coex/run.py`): K = 3 competing explanations, n0 = 3 initial documents per probe,
window W = 20, retained h = 10, T = 9 reranker calls, shared-evidence slot share 0.5, UCB coefficient c = 0.3,
reward update rate 0.5, kNN seed threshold tau = 0.7 (up to 3 seeds per probe, up to half of a probe's slots),
probe revision after rounds 2, 4 and 6, at least 4 new documents per round. All LLM calls use temperature 0.

## Evaluation

```bash
python eval/eval_bright.py --results results/coex
python eval/eval_bright_pro.py --results results/coex --bright_pro_repo external/Bright-Pro
```

BRIGHT: nDCG@10 with excluded documents removed (official protocol), macro-averaged over domains.
BRIGHT-Pro: alpha-nDCG@10 (alpha = 0.5) and A-Recall@10 with the official BRIGHT-Pro functions, over queries with
gold documents, macro-averaged over the 7 domains.

LLM outputs served by vLLM are not bit-exact across hardware and versions, so rerunning gives numbers close to, but
not identical with, those in the paper.

## Acknowledgements

BRIGHT and BRIGHT-Pro are used under their licenses (CC BY 4.0). The reranking prompt and the document-graph format
follow GAREN.
