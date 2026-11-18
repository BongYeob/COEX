"""BM25 identical to the official BRIGHT implementation (xlang-ai/BRIGHT, retrievers.py: retrieval_bm25).

Tokenization: Pyserini Lucene English analyzer (lowercasing, stopwords, Porter stemming).
Scoring: gensim LuceneBM25Model (k1=0.9, b=0.4) -> SparseMatrixSimilarity without normalization.
Pyserini requires Java 11 (set JAVA_HOME / JVM_PATH if it is not found automatically).
The analyzed corpus is cached in <cache_dir>/<task>.json.
"""
import json
import os

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity
from pyserini import analysis

_ANALYZER = analysis.Analyzer(analysis.get_lucene_analyzer())


def analyze(text):
    return _ANALYZER.analyze(text)


class BrightBM25:
    def __init__(self, task, docs, cache_dir="cache/bm25"):
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{task}.json")
        self.ids = [d["id"] for d in docs]
        corpus = None
        if os.path.exists(path):
            cached = json.load(open(path))
            if cached.get("ids") == self.ids:
                corpus = cached["tokens"]
        if corpus is None:
            corpus = [analyze(d["content"]) for d in docs]
            json.dump({"ids": self.ids, "tokens": corpus}, open(path, "w"))
        self.dictionary = Dictionary(corpus)
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=0.9, b=0.4)
        bm25_corpus = self.model[list(map(self.dictionary.doc2bow, corpus))]
        self.index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(self.dictionary),
                                            normalize_queries=False, normalize_documents=False)

    def scores(self, text):
        q = self.model[self.dictionary.doc2bow(analyze(text))]
        return np.asarray(self.index[q], dtype=np.float32)

    def search(self, text, k=1000, exclude=()):
        sc = self.scores(text)
        out = []
        for i in np.argsort(-sc):
            if self.ids[i] in exclude:
                continue
            out.append(self.ids[i])
            if len(out) >= k:
                break
        return out
