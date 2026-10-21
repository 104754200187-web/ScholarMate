"""Offline evaluation harness for ScholarMate's retrieval stack.

Measures retrieval Recall@K / Hits@K against hand-picked gold papers, and
optionally replays stored snapshots to summarize claim/citation verdicts.
Every metric call is deterministic on the same date/filters, so changes to the
pipeline can be regression-tested offline.

Dataset format (JSONL, one object per line):
    {"question": "…", "gold": ["gold paper title", "…"]}

Usage:
    python3 eval.py                      # run eval_data/gold.jsonl if present
    python3 eval.py --data my.jsonl --limit 5
    python3 eval.py --mode snapshot --snapshots outputs/history
"""

import argparse
import glob
import json
import os
import re
import time

import graph
import search


def _norm(title):
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def retrieval_metrics(question, gold_titles, window_years=0, review_only=False,
                      limit=3):
    """Plan queries (DeepSeek), retrieve from all enabled sources, then compute
    recall/precision of the union pool against the gold list."""
    gold = {_norm(t) for t in gold_titles if t}
    filters = None
    if window_years:
        filters = {"min_year": 2026 - window_years, "review_only": review_only}
    queries = graph.plan_queries(question)
    details = search.search_many_detailed(
        queries, limit=8, filters=filters
    )
    pool, source_hits = {}, {}
    for d in details:
        for s in d.get("sources", []):
            source_hits[s["source"]] = source_hits.get(s["source"], 0) + s["hits"]
        for p in d["papers"]:
            if not p.get("abstract"):
                continue
            pool[_norm(p["title"])] = p
    ranked = sorted(pool.values(), key=lambda p: -(p.get("citations") or 0))
    found = {_norm(p["title"]) for p in ranked[:20]} & gold
    total = len(pool)
    return {
        "question": question,
        "queries": queries,
        "pool_size": total,
        "source_hits": source_hits,
        "gold_total": len(gold),
        "recall_at_20": len(found) / len(gold) if gold else None,
        "found_gold": sorted(found),
        "missing_gold": sorted(gold - found),
    }


def summarize_snapshots(path):
    """Aggregate stored run snapshots into citation/claim quality numbers."""
    rows = []
    for fp in sorted(glob.glob(os.path.join(path, "*.json"))):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("mode") != "ask":
            continue
        rules = data.get("audits", {}).get("rules", {}).get("items", [])
        llm = data.get("audits", {}).get("llm", {}).get("verdicts", [])
        claims = data.get("audits", {}).get("claims", {}).get("verdicts", [])
        def unjudged(item):
            return (item.get("judged") is False or item.get("badge") == "unjudged"
                    or "supported" not in item)
        rows.append({
            "id": data.get("id"),
            "rule_bad": sum(1 for i in rules if i.get("status") == "bad"),
            "llm_unsupported": sum(1 for v in llm
                                   if not v.get("supported") and not unjudged(v)),
            "llm_unjudged": sum(1 for v in llm if unjudged(v)),
            "llm_total": len(llm),
            "claims_unsupported": sum(1 for c in claims
                                      if not c.get("supported") and not unjudged(c)),
            "claims_unjudged": sum(1 for c in claims if unjudged(c)),
            "claims_total": len(claims),
        })
    if not rows:
        return {"snapshots": 0}
    def avg(key):
        vals = [r[key] for r in rows]
        return round(sum(vals) / len(vals), 2)
    return {
        "snapshots": len(rows),
        "avg_rule_bad": avg("rule_bad"),
        "avg_llm_unsupported": avg("llm_unsupported"),
        "avg_llm_unjudged": avg("llm_unjudged"),
        "avg_llm_total": avg("llm_total"),
        "avg_claims_unsupported": avg("claims_unsupported"),
        "avg_claims_unjudged": avg("claims_unjudged"),
        "avg_claims_total": avg("claims_total"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="eval_data/gold.jsonl")
    ap.add_argument("--mode", choices=["retrieval", "snapshot"], default="retrieval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--years", type=int, default=0, help="restrict to last N years")
    ap.add_argument("--review", action="store_true", help="only survey/review papers")
    ap.add_argument("--snapshots", default="outputs/history")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.mode == "snapshot":
        summary = summarize_snapshots(args.snapshots)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if not os.path.exists(args.data):
        print(f"eval: dataset not found: {args.data}")
        return
    results = []
    with open(args.data) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    for i, entry in enumerate(entries):
        if args.limit and i >= args.limit:
            break
        t0 = time.time()
        metric = retrieval_metrics(
            entry["question"], entry.get("gold", []),
            window_years=args.years, review_only=args.review,
        )
        metric["seconds"] = round(time.time() - t0, 1)
        results.append(metric)
        print(f"[{i + 1}/{len(entries)}] recall@20={metric['recall_at_20']} "
              f"pool={metric['pool_size']} ({metric['seconds']}s)")

    agg = {}
    recalls = [r["recall_at_20"] for r in results if r["recall_at_20"] is not None]
    if recalls:
        agg["mean_recall_at_20"] = round(sum(recalls) / len(recalls), 3)
    out_path = args.out or os.path.join(
        "outputs", f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump({"summary": agg, "results": results},
              open(out_path, "w"), ensure_ascii=False, indent=2)
    print("summary:", json.dumps(agg, ensure_ascii=False))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
