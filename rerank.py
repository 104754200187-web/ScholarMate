"""LLM-based reranking (pure stdlib).

DeepSeek scores candidates by relevance: paper pool vs. the research question
(used before generation), and BM25 full-text passages vs. a claim (used before
the per-claim judge). Falls back to the original order when the LLM call or
parsing fails. Disable entirely with RERANK=0.
"""

import os
import re

import llm

ENABLED = os.environ.get("RERANK", "1") != "0"
BATCH_SIZE = 10
PASSAGE_MAX_CHARS = 300

PAPER_PROMPT = """You are reranking search results for a research question.
Score the relevance of each candidate paper from 0 to 10 (10 = directly
answers the question, 0 = irrelevant). Be strict: only strongly on-topic
papers score above 7. Reply ONLY with JSON:
{"scores": [{"id": <int>, "score": <float>}]}
Include every candidate id exactly once."""

PASSAGE_PROMPT = """You are reranking evidence passages for a claim.
Score each passage from 0 to 10 by how directly it supports the claim.
Reply ONLY with JSON:
{"scores": [{"id": <int>, "score": <float>}]}
Include every candidate id exactly once. Be strict: score above 7 only for
passages that directly back the claim."""


def _parse_scores(content, expected_ids):
    """Tolerantly parse {"scores": [{"id": n, "score": s}]} replies."""
    try:
        data = llm.parse_json(content)
    except ValueError:
        data = None
    if isinstance(data, dict):
        items = data.get("scores", [])
    elif isinstance(data, list):
        items = data
    else:
        items = []
    scores = {}
    for item in items:
        try:
            scores[int(item["id"])] = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
    if not scores:  # regex fallback for malformed JSON
        for m in re.finditer(r'"id"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*([\d.]+)', content):
            scores[int(m.group(1))] = float(m.group(2))
    return [(i, scores.get(i)) for i in expected_ids]


def rerank_papers(question, papers, top_k=None, batch_size=BATCH_SIZE):
    """Score papers by relevance to the question. Returns a list of
    (paper, score) sorted by score desc, or None on total failure (caller
    keeps the original order)."""
    if not ENABLED or not papers:
        return None
    items = list(enumerate(papers))
    ranked = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        blocks = []
        for idx, p in batch:
            abstract = (p.get("abstract") or "")[:300]
            blocks.append(
                f"[{idx}] {p.get('title', '')} ({p.get('year', '')})\n"
                f"Abstract: {abstract}"
            )
        try:
            content = llm.chat(
                [
                    {"role": "system", "content": PAPER_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\nCandidates:\n"
                            + "\n\n".join(blocks)
                        ),
                    },
                ],
                max_tokens=512,
                temperature=0,
            )
        except Exception as e:
            print(f"      [rerank] paper scoring failed ({e}); keeping original order")
            return None
        scored = _parse_scores(content, [idx for idx, _ in batch])
        ranked.extend((idx, s if s is not None else 0.0) for idx, s in scored)
    ranked.sort(key=lambda x: (-x[1], x[0]))
    if top_k:
        ranked = ranked[:top_k]
    return [(papers[idx], score) for idx, score in ranked]


def rerank_passages(query, passages, top_k=None):
    """Score passages by relevance to a claim/query. Returns top_k passage
    dicts (with a 'score' key) or None on total failure (caller keeps the
    BM25 order)."""
    if not ENABLED or not passages:
        return None
    blocks = [
        f"[{i}] {p['text'][:PASSAGE_MAX_CHARS]}" for i, p in enumerate(passages)
    ]
    try:
        content = llm.chat(
            [
                {"role": "system", "content": PASSAGE_PROMPT},
                {
                    "role": "user",
                    "content": f"Claim: {query}\n\nPassages:\n" + "\n\n".join(blocks),
                },
            ],
            max_tokens=512,
            temperature=0,
        )
    except Exception as e:
        print(f"      [rerank] passage scoring failed ({e}); keeping BM25 order")
        return None
    scored = _parse_scores(content, list(range(len(passages))))
    if not any(s is not None for _, s in scored):
        return None
    ranked = sorted(
        ((i, s if s is not None else 0.0) for i, s in scored),
        key=lambda x: (-x[1], x[0]),
    )
    if top_k:
        ranked = ranked[:top_k]
    return [{**passages[i], "score": round(score, 2)} for i, score in ranked]


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="LLM rerank demo")
    ap.add_argument("question")
    ap.add_argument("papers_json", help="path to a JSON list of {title, abstract, year}")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    with open(args.papers_json, encoding="utf-8") as f:
        papers = json.load(f)
    result = rerank_papers(args.question, papers, top_k=args.top)
    for i, (p, score) in enumerate(result, 1):
        print(f"{i}. [{score:.1f}] {p.get('title')} ({p.get('year')})")
