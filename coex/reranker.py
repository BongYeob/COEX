"""Listwise LLM reranker R (vLLM OpenAI-compatible server).

Uses the shared reasoning-based reranking prompt released with GAREN (reasoning trace, then a ranking of at least
10 documents). Settings: documents truncated to 512 tokens, query to 1024 tokens, ftfy normalization,
temperature 0, top_p 1, seed 42, max_tokens 2048. Only the numbers after "[Document Ranking]" are read as the
ranking; documents missing from the output are appended in input order, and the number of documents actually
listed is kept in ``n_listed`` (unlisted documents receive a tied score in the judgment loop).
"""
import asyncio

from ftfy import fix_text
from openai import AsyncOpenAI
from transformers import AutoTokenizer

SYSTEM_REASONING = (
    "You are an AI assistant that analyzes complex questions and identifies which "
    "documents best support answering them.\n\n"
    "Given a user's query and a set of documents, your task is to:\n"
    "1. Generate a reasoning trace, thinking step by step about what knowledge or "
    "types of information are necessary to answer the query. These should be "
    "abstract but specific enough to guide document selection.\n"
    "2. Select and rank at least 10 documents that best support the reasoning "
    "steps. Order them from most to least useful using `>` between document IDs "
    "(e.g., [3] > [7]).\n\n"
    "Use the following format:\n"
    "[Reasoning Trace]\n"
    "Step 1: <First reasoning step>\n"
    "Step 2: <Second reasoning step>\n"
    "...\n"
    "Step N: <Final reasoning step>\n\n"
    "[Document Ranking]\n"
    "[9] > [5] > [6] > ... > [12]\n\n"
    "Only produce the output in the format shown above."
)


def parse_ranking(text, n):
    """Returns (full order of indices, number of indices actually listed)."""
    text = (text or "").split("</think>")[-1].split("[Document Ranking]")[-1]
    s = "".join(c if c.isdigit() else " " for c in text)
    out = []
    for x in s.split():
        i = int(x) - 1
        if 0 <= i < n and i not in out:
            out.append(i)
    return out + [i for i in range(n) if i not in out], len(out)


class ListwiseReranker:
    def __init__(self, url, model, max_dlen=512, max_qlen=1024, max_tokens=2048, concurrency=48, seed=42):
        self.url, self.model = url, model
        self.max_dlen, self.max_qlen, self.max_tokens, self.concurrency, self.seed = max_dlen, max_qlen, max_tokens, concurrency, seed
        self.tok = AutoTokenizer.from_pretrained(model)
        self.n_listed = []

    def truncate(self, text, length):
        return self.tok.convert_tokens_to_string(self.tok.tokenize(text)[:length])

    def messages(self, query, contents):
        docs = "\n".join(f"[{i + 1}]: {self.truncate(c.strip(), self.max_dlen)}" for i, c in enumerate(contents)).strip()
        q = self.truncate(query, self.max_qlen).strip()
        return [{"role": "system", "content": SYSTEM_REASONING},
                {"role": "user", "content": fix_text(f"[Query]\n{q}\n\n[Documents]\n{docs}\n\n")}]

    def rank(self, items):
        """items: [(query, [doc_id], [content])] -> [[doc_id in ranked order]]; also sets self.n_listed."""
        async def run():
            client = AsyncOpenAI(base_url=self.url, api_key="x")
            sem = asyncio.Semaphore(self.concurrency)

            async def one(q, ids, contents):
                msgs = self.messages(q, contents)
                async with sem:
                    for _ in range(3):
                        try:
                            r = await client.chat.completions.create(model=self.model, messages=msgs, max_tokens=self.max_tokens,
                                                                     temperature=0.0, top_p=1.0, seed=self.seed)
                            order, k = parse_ranking(r.choices[0].message.content or "", len(ids))
                            return [ids[i] for i in order], k
                        except Exception:
                            await asyncio.sleep(1)
                return list(ids), 0
            try:
                res = await asyncio.gather(*[one(*it) for it in items])
                self.n_listed = [k for _, k in res]
                return [o for o, _ in res]
            finally:
                await client.close()
        return asyncio.run(run())
