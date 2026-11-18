"""COEX: explanation-guided navigation with a judgment loop.

Probes (search targets)
  - K competing explanations generated from the question (INIT_PROMPT),
  - one shared-evidence probe (knowledge any correct answer needs; SHARED_PROMPT, only "shared_evidence" is used),
  - the original query (its search text never changes).
  Each probe ranks the whole corpus with BM25 on its own search text.

Judgment loop (window W=20, retained h=10, T=9 reranker calls)
  round 1 : the first n0=3 unjudged documents of every probe, filled up to W with original-query BM25.
  each later round t:
    1) accumulated score S(d): mean over all judgments of the position score 1 - i/(|w|-1); documents the
       reranker did not list share the mean score of the remaining positions.
    2) re-evaluation before promotion: documents judged for the first time in round t-1 that entered the
       window top-h are re-judged in round t; the h best of the other judged documents (by S) are retained.
    3) the remaining (>= 4) slots take new documents: half of them (rounded) go to the shared-evidence probe,
       the rest are allocated among the explanation and original-query probes by UCB  Q_p + c*sqrt(ln(t+1)/(1+n_p)).
       Within a probe, up to half of its slots are filled with kNN neighbours of up to 3 of its documents with
       S >= tau (0.7); the rest are its next unjudged BM25 documents.
    4) reward: r_t(p) = mean position score of the new documents probe p supplied, Q_p <- 0.5 Q_p + 0.5 r_t(p).
    5) probe update (one LLM call per query, GAP_PROMPT): knowledge still missing for each explanation and for
       any answer; explanation / shared probes then search with "missing + key knowledge". After rounds 2, 4 and 6
       the explanations may also be kept, refined, dropped or replaced (GAP_REVISE).
  last round: no new documents; the window is re-judged only.
  final ranking: all judged documents by S(d), then the remaining original-query BM25 documents.

Usage:
  python -m coex.run --task biology --url http://localhost:8000/v1 --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
"""
import argparse
import asyncio
import json
import math
import os
import re
from collections import OrderedDict, defaultdict

import numpy as np
from openai import AsyncOpenAI

from .bm25 import BrightBM25
from .data import load_task
from .reranker import ListwiseReranker

INIT_PROMPT = """Question:
{query}

Propose {k} DISTINCT, competing explanations or solution strategies for this question. Different explanations should rely on different background knowledge (different principles, mechanisms, theorems, or language features). For each, give a one-sentence description and the key knowledge (canonical names) a document would need to contain to support it.
Return ONLY a JSON list: [{{"hypothesis": "...", "key_knowledge": ["...", "..."]}}, ...]"""

SHARED_PROMPT = """Question:
{query}
{corpus}
Think about how this question could be answered.
1. "shared_evidence": knowledge that ANY correct answer will need regardless of which explanation or strategy is right (definitions, basic facts, and - if the question requires a specific programming language, tool or setting - the concrete features, syntax or APIs of that setting needed to write the answer).
2. "explanations": {k} DISTINCT, competing explanations or solution strategies. For each, give a one-sentence description and the "evidence": what a supporting document in a reference collection would need to explain, stated concretely within the constraints of the question.
Use short, concrete phrases (lowercase).
Return ONLY JSON: {{"shared_evidence": ["...", ...], "explanations": [{{"hypothesis": "...", "evidence": ["...", ...]}}, ...]}}"""

GAP_PROMPT = """Question:
{query}

Top documents found so far:
{docs}

Candidate explanations / solution strategies (with evidence score 0-1):
{hyps}

Already searched for: {explored}

{instruction}
For each explanation, list up to 4 pieces of knowledge that are still NEEDED to fully answer the question if that explanation is correct, but are NOT yet covered by the top documents. Also list up to 4 pieces of knowledge that are still needed by ANY correct answer ("shared_missing"). These can be concepts, mechanisms, principles, theorems, or for programming questions the language features, syntax, standard library APIs or idioms needed in the required language. Use short canonical names (lowercase). Prefer items not already searched for.
Return ONLY JSON: {{"explanations": [{{"from": <index>, "hypothesis": "...", "missing": ["...", ...]}}, ...], "shared_missing": ["...", ...]}}"""

GAP_KEEP = "Keep the explanations unchanged (same index and text)."
GAP_REVISE = ("You may also revise the explanations: keep or refine those with strong evidence, drop those with weak evidence, "
              "and add new, different ones (from = -1) if all evidence is weak. At most {k} explanations.")

SHARED_LABEL = "Shared prerequisite knowledge needed by any correct answer"


def jobj(text, open_="{", close_="}"):
    m = re.search(re.escape(open_) + r".*" + re.escape(close_), text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def clean(xs, n=None):
    out = [x.strip().lower() for x in (xs or []) if isinstance(x, str) and x.strip()]
    return out[:n] if n else out


def llm_batch(url, model, prompts, max_tokens=900, concurrency=48):
    async def run():
        client = AsyncOpenAI(base_url=url, api_key="x")
        sem = asyncio.Semaphore(concurrency)

        async def one(p):
            async with sem:
                for _ in range(3):
                    try:
                        r = await client.chat.completions.create(model=model, temperature=0.0, max_tokens=max_tokens,
                                                                 messages=[{"role": "user", "content": p}])
                        return r.choices[0].message.content or ""
                    except Exception:
                        await asyncio.sleep(1)
            return ""
        try:
            return await asyncio.gather(*[one(p) for p in prompts])
        finally:
            await client.close()
    return asyncio.run(run())


class Probe:
    """One search target: a competing explanation, the shared-evidence probe, or the original query."""

    def __init__(self, hypothesis, knowledge, shared=False, support=None, Q=None, orig=False):
        self.hypothesis, self.knowledge, self.shared, self.orig = hypothesis, list(knowledge), shared, orig
        self.missing, self.support, self.Q = [], set(support or []), Q
        self.n, self.last, self.rank, self.ptr = 0, [], [], 0

    def text(self):
        if self.orig:
            return self.hypothesis
        if self.missing:
            return " ".join(self.missing) + " " + " ".join(self.knowledge)
        return self.hypothesis + " " + " ".join(self.knowledge)


class COEX:
    def __init__(self, bm25, reranker, url, model, knn=None, K=3, n0=3, m_top=3, ucb_c=0.3, tau=0.7, alpha=0.5,
                 shared_share=0.5, revise_at=(2, 4, 6), window=20, top_keep=10, n_iters=9, doc_words=100, min_new=4):
        self.bm, self.rr, self.url, self.model, self.knn = bm25, reranker, url, model, knn
        self.K, self.n0, self.m_top, self.ucb_c, self.tau, self.alpha = K, n0, m_top, ucb_c, tau, alpha
        self.shared_share, self.revise_at = shared_share, set(revise_at)
        self.window, self.top_keep, self.n_iters, self.doc_words, self.min_new = window, top_keep, n_iters, doc_words, min_new

    def search(self, p, k=300):
        sc = self.bm.scores(p.text())
        idx = np.argpartition(-sc, k)[:k]
        p.rank, p.ptr = [self.bm.ids[i] for i in idx[np.argsort(-sc[idx])]], 0

    @staticmethod
    def score(st, d):
        s = st["pos"].get(d)
        return (sum(s) / len(s), len(s)) if s else (-1.0, 0)

    def belief(self, st, p):
        js = sorted((self.score(st, d)[0] for d in p.support if d in st["pos"]), reverse=True)[:self.m_top]
        return float(np.mean(js)) if js else 0.5

    @staticmethod
    def take(st, p, n, taken):
        out = []
        while len(out) < n and p.ptr < len(p.rank):
            d = p.rank[p.ptr]; p.ptr += 1
            if d not in st["seen"] and d not in st["q"]["excluded"] and d not in taken:
                out.append(d)
        return out

    def run(self, queries, content, init, shared):
        S = {}
        for q in queries:
            probes = [Probe(h["hypothesis"], h.get("key_knowledge", [])) for h in init[q["id"]]][:self.K] or [Probe(q["query"], [])]
            probes.append(Probe(q["query"], [], orig=True))
            if shared.get(q["id"]):
                probes.append(Probe(SHARED_LABEL, shared[q["id"]], shared=True))
            for p in probes:
                self.search(p)
            S[q["id"]] = {"q": q, "probes": probes, "pool": self.bm.search(q["query"], 1000, q["excluded"]),
                          "seen": OrderedDict(), "pos": defaultdict(list), "explored": [], "prov": [], "new": [],
                          "trace": {"probes": [], "Q": [], "windows": [], "listed": []}}

        for it in range(self.n_iters):
            items = []
            for st in S.values():
                if it == 0:
                    win = []
                    for p in st["probes"]:
                        got = self.take(st, p, self.n0, set(win))
                        p.support.update(got); p.last = got; win += got
                else:
                    win = self.next_window(st, it)
                for d in st["pool"]:                      # remaining slots: original-query BM25
                    if len(win) >= self.window:
                        break
                    if d not in win and d not in st["seen"]:
                        win.append(d)
                st["new"] = [d for d in win if d not in st["seen"]]
                st["seen"].update(dict.fromkeys(win))
                items.append((st["q"]["query"], win, [content[d] for d in win]))
            ranked = self.rr.rank(items)
            for qi, (st, r) in enumerate(zip(S.values(), ranked)):
                n = max(1, len(r) - 1)
                k = self.rr.n_listed[qi]
                k = len(r) if k <= 0 else k               # parse failure: keep the input order
                tie = float(np.mean([1 - i / n for i in range(k, len(r))])) if k < len(r) else 0.0
                wpos = {}
                for i, d in enumerate(r):
                    v = 1 - i / n if i < k else tie
                    st["pos"][d].append(v); wpos[d] = v
                ridx = {d: i for i, d in enumerate(r)}
                for p in st["probes"]:                     # reward of each probe that supplied new documents
                    if p.last:
                        rw = float(np.mean([wpos.get(d, 0.0) for d in p.last]))
                        p.Q = rw if p.Q is None else (1 - self.alpha) * p.Q + self.alpha * rw
                        p.last = []
                # provisional: first judged in this round and placed in the window top-h -> re-judged next round
                st["prov"] = [d for d in st["new"] if ridx.get(d, len(r)) < min(k, self.top_keep)] if it > 0 else []
                st["trace"]["windows"].append(list(r))
                st["trace"]["listed"].append(k)
                st["trace"]["Q"].append([None if p.Q is None else round(p.Q, 3) for p in st["probes"]])
                st["trace"]["probes"].append([{"h": p.hypothesis, "missing": p.missing, "shared": p.shared, "orig": p.orig,
                                               "belief": round(self.belief(st, p), 3)} for p in st["probes"]])
            if it == self.n_iters - 1:
                break
            self.update_probes(S, content, revise=(it + 1) in self.revise_at)
            print(f"#> [COEX] round {it + 1} done", flush=True)

        final, traces = {}, {}
        for qid, st in S.items():
            s = sorted(st["seen"], key=lambda d: self.score(st, d), reverse=True)
            final[qid] = s + [d for d in st["pool"] if d not in st["seen"]]
            traces[qid] = st["trace"]
        return final, traces

    def next_window(self, st, it):
        """Retain the top-h judged documents (excluding provisional ones), re-judge the provisional documents,
        and fill the rest with new documents (at least min_new). In the last round there are no new documents."""
        last = it == self.n_iters - 1
        prov = sorted(st["prov"], key=lambda d: self.score(st, d), reverse=True)
        ps = set(prov)
        ranked = [d for d in sorted(st["seen"], key=lambda d: self.score(st, d), reverse=True) if d not in ps]
        keep = ranked[:self.top_keep]
        room = self.window - len(keep)
        prov = prov[:max(0, room - (0 if last else self.min_new))]
        n_new = 0 if last else room - len(prov)
        keep = keep + prov
        if last:
            keep += ranked[self.top_keep:self.top_keep + room - len(prov)]
        if n_new <= 0:
            return keep
        return keep + self.new_documents(st, it, keep, n_new)

    def new_documents(self, st, it, keep, n_new):
        probes = st["probes"]
        sc = [0.5 if p.Q is None else p.Q for p in probes]
        alloc, virt = [0] * len(probes), [p.n for p in probes]
        sh = [i for i, p in enumerate(probes) if p.shared]
        free = n_new
        if sh and self.shared_share > 0:              # guaranteed share for the shared-evidence probe
            k = int(round(n_new * self.shared_share)); alloc[sh[0]] += k; free -= k
        cand = [i for i, p in enumerate(probes) if not (self.shared_share > 0 and p.shared)] or list(range(len(probes)))
        for _ in range(free):                          # UCB allocation of the remaining slots
            u = {i: sc[i] + self.ucb_c * math.sqrt(math.log(it + 2) / (1 + virt[i])) for i in cand}
            j = max(u, key=u.get); alloc[j] += 1; virt[j] += 1
        out = []
        for p, n in zip(probes, alloc):
            if n == 0:
                continue
            got = []
            if self.knn is not None:                    # kNN fill: neighbours of the probe's strong documents
                strong = [d for d in sorted(p.support, key=lambda d: self.score(st, d), reverse=True) if self.score(st, d)[0] >= self.tau][:3]
                for d in strong:
                    for x in self.knn(d):
                        if len(got) >= n // 2:
                            break
                        if x not in st["seen"] and x not in st["q"]["excluded"] and x not in out and x not in got and x not in keep:
                            got.append(x)
            got += self.take(st, p, n - len(got), set(out) | set(got) | set(keep))
            p.support.update(got); p.n += len(got); p.last = list(got)
            out += got
        for p in probes:                                # a document high in another probe's BM25 list also supports it
            top = set(p.rank[:max(50, p.ptr)])
            p.support.update(d for d in out if d in top)
        return out[:n_new]

    def update_probes(self, S, content, revise):
        prompts = []
        for st in S.values():
            top = sorted(st["pos"], key=lambda d: self.score(st, d), reverse=True)[:5]
            docs = "\n".join(f"[{i + 1}] " + " ".join(content[d].split()[:self.doc_words]) for i, d in enumerate(top))
            comp = [p for p in st["probes"] if not p.shared and not p.orig]
            lines = "\n".join(f"[{i}] score={self.belief(st, p):.2f} :: {p.hypothesis}" for i, p in enumerate(comp))
            prompts.append(GAP_PROMPT.format(query=st["q"]["query"], docs=docs or "(none)", hyps=lines,
                                             explored=", ".join(st["explored"][-40:]) or "(none)",
                                             instruction=GAP_REVISE.format(k=self.K) if revise else GAP_KEEP))
        outs = llm_batch(self.url, self.model, prompts, max_tokens=900)
        for st, o in zip(S.values(), outs):
            obj = jobj(o) or {}
            rows = [x for x in obj.get("explanations", []) if isinstance(x, dict)]
            comp = [p for p in st["probes"] if not p.shared and not p.orig]
            fixed = [p for p in st["probes"] if p.shared]
            origs = [p for p in st["probes"] if p.orig]
            qs = [p.Q for p in st["probes"] if p.Q is not None]
            if revise and rows:                         # keep / refine / drop / add explanations
                new = []
                for r in rows[:self.K]:
                    src = r.get("from", -1)
                    ok = isinstance(src, int) and 0 <= src < len(comp)
                    hyp = r.get("hypothesis") or (comp[src].hypothesis if ok else "")
                    if not hyp:
                        continue
                    p = Probe(hyp, comp[src].knowledge if ok else [], support=comp[src].support if ok else None,
                              Q=comp[src].Q if ok else (float(np.mean(qs)) if qs else None))
                    p.missing = clean(r.get("missing"), 4)
                    new.append(p)
                comp = new or comp
            else:
                for r in rows:
                    src = r.get("from", -1)
                    if isinstance(src, int) and 0 <= src < len(comp):
                        comp[src].missing = clean(r.get("missing"), 4)
            sm = clean(obj.get("shared_missing"), 4)
            for p in fixed:
                if sm:
                    p.missing = sm
            for p in comp + fixed:
                if p.missing:
                    st["explored"].extend(x for x in p.missing if x not in st["explored"])
                    self.search(p)
            st["probes"] = comp + origs + fixed


def load_knn(graph_dir, task):
    """Document kNN graph (k=16, Qwen3-Embedding-4B, exact cosine) in the format released with GAREN."""
    import npids
    gd = os.path.join(graph_dir, task)
    meta = json.load(open(os.path.join(gd, "pt_meta.json")))
    adj = np.fromfile(os.path.join(gd, "edges.u32.np"), dtype=np.uint32).reshape(meta["doc_count"], meta["k"])
    dn = npids.Lookup(os.path.join(gd, "docnos.npids"))
    ids = [str(dn[i]) for i in range(meta["doc_count"])]
    idx = {d: i for i, d in enumerate(ids)}
    return lambda d: [] if d not in idx else [ids[int(j)] for j in adj[idx[d]]]


def initial_probes(task, url, model, K, shared_K, cache_dir, bm25_cache):
    """Competing explanations (INIT_PROMPT) and shared evidence (SHARED_PROMPT, which also shows the first 60 words of
    the top-5 original-query BM25 documents). Generated once per BRIGHT domain and cached; a BRIGHT-Pro task
    (bp_<domain>, same query ids) reuses the cache of its BRIGHT domain."""
    base = task[3:] if task.startswith("bp_") else task
    os.makedirs(cache_dir, exist_ok=True)
    ip, sp = os.path.join(cache_dir, f"{base}_hyps_k{K}.json"), os.path.join(cache_dir, f"{base}_shared_k{shared_K}.json")
    ic = json.load(open(ip)) if os.path.exists(ip) else {}
    scache = json.load(open(sp)) if os.path.exists(sp) else {}
    docs, queries = load_task(base)
    queries = [q for q in queries if q["gold"]]
    need = [q for q in queries if q["id"] not in ic]
    if need:
        for q, o in zip(need, llm_batch(url, model, [INIT_PROMPT.format(query=q["query"], k=K) for q in need])):
            ic[q["id"]] = {"hyps": jobj(o, "[", "]") or []}
        json.dump(ic, open(ip, "w"), indent=1, ensure_ascii=False)
    need = [q for q in queries if q["id"] not in scache]
    if need:
        content = {d["id"]: d["content"] for d in docs}
        bm = BrightBM25(base, docs, bm25_cache)
        prompts = []
        for q in need:
            top = [" ".join(content[d].split()[:60]) for d in bm.search(q["query"], 5, q["excluded"])]
            corp = ("\nFor context, here are the beginnings of a few documents from the reference collection that will be searched:\n"
                    + "\n".join(f"- {d}" for d in top) + "\n")
            prompts.append(SHARED_PROMPT.format(query=q["query"], corpus=corp, k=shared_K))
        for q, o in zip(need, llm_batch(url, model, prompts)):
            scache[q["id"]] = jobj(o) or {}
        json.dump(scache, open(sp, "w"), indent=1, ensure_ascii=False)
    init = {}
    for qid, v in ic.items():
        hs = [{"hypothesis": h.get("hypothesis") or h.get("hypothsis") or "", "key_knowledge": clean(h.get("key_knowledge"))}
              for h in v.get("hyps", []) if isinstance(h, dict)]
        init[qid] = [h for h in hs if h["hypothesis"]]
    shared = {qid: clean(v.get("shared_evidence")) for qid, v in scache.items()}
    return init, shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="BRIGHT domain (e.g. biology) or BRIGHT-Pro domain (e.g. bp_biology)")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8")
    ap.add_argument("--K", type=int, default=3, help="number of competing explanations")
    ap.add_argument("--shared_K", type=int, default=4, help="k in SHARED_PROMPT (its explanations are not used)")
    ap.add_argument("--graph_dir", default="data/GRAPH/qwen3/graph")
    ap.add_argument("--cache_dir", default="cache/probes")
    ap.add_argument("--bm25_cache", default="cache/bm25")
    ap.add_argument("--out_dir", default="results/coex")
    ap.add_argument("--max_queries", type=int, default=0)
    a = ap.parse_args()

    docs, queries = load_task(a.task)
    queries = [q for q in queries if q["gold"]][:a.max_queries or None]
    content = {d["id"]: d["content"] for d in docs}
    bm = BrightBM25(a.task, docs, a.bm25_cache)
    init, shared = initial_probes(a.task, a.url, a.model, a.K, a.shared_K, a.cache_dir, a.bm25_cache)
    nav = COEX(bm, ListwiseReranker(a.url, a.model), a.url, a.model, knn=load_knn(a.graph_dir, a.task), K=a.K)
    print(f"#> [COEX] task={a.task} queries={len(queries)}", flush=True)
    final, traces = nav.run(queries, content, init, shared)

    out = os.path.join(a.out_dir, a.task)
    os.makedirs(out, exist_ok=True)
    json.dump({k: v[:1000] for k, v in final.items()}, open(os.path.join(out, "ranking.json"), "w"))
    json.dump(traces, open(os.path.join(out, "trace.json"), "w"))
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"#> saved {out}/ranking.json")


if __name__ == "__main__":
    main()
