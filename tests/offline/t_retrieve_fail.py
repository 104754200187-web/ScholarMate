"""A run where every source errored must fail loudly, not return an empty list."""
import sys
import _paths  # noqa: F401
import graph, search

fails = []
def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + repr(extra)))
    if not cond: fails.append(name)

def item(err, papers=None, query="q", sources=None):
    return {"query": query, "papers": papers or [], "sources": sources or [],
            "error": err}

def state(queries=("q",), rounds=0):
    return {"rounds": rounds, "queries": list(queries), "pool": {}, "filters": None,
            "retrieval_log": []}

def run(main, classic, st=None):
    real_m, real_c = search.search_many_detailed, search.search_classics
    search.search_many_detailed, search.search_classics = main, classic
    try:
        return graph.node_retrieve(st or state())
    finally:
        search.search_many_detailed, search.search_classics = real_m, real_c

REFUSED = "semantic_scholar: connection failed: [Errno 111] Connection refused"

# 1. the 2026-10-19 case: every source refused, both passes empty
try:
    run(lambda *a, **k: [item(REFUSED)], lambda *a, **k: [item("openalex: " + REFUSED)])
    check("every source failing raises instead of finishing empty", False, "no exception")
except RuntimeError as e:
    msg = str(e)
    check("every source failing raises instead of finishing empty", True)
    check("the message says this is not 'no relevant papers'",
          "不是「没有相关文献」" in msg, msg)
    check("it carries the underlying error", "Connection refused" in msg, msg)
    check("it points at the proxy settings a reader can act on",
          "HTTP_PROXY" in msg and "HTTPS_PROXY" in msg, msg)

# 2. a source answered, it simply matched nothing: not an error
out = run(lambda *a, **k: [item(None)], lambda *a, **k: [item(None)])
check("a genuine zero-result answer still completes", isinstance(out, dict)
      and out["pool"] == {}, out)

# 3. one query answered while another failed: not an error (partial outage)
out = run(lambda *a, **k: [item(REFUSED, query="q1"), item(None, query="q2")],
          lambda *a, **k: [item(None)])
check("a partial source outage does not fail the run", isinstance(out, dict), out)

# 4. classic pass rescued the round after the main pass failed
paper = {"title": "A paper", "abstract": "not empty", "year": 2020, "citations": 5,
         "url": "https://x", "authors": [], "source": "openalex", "pdf_url": "",
         "arxiv_id": "", "doi": "", "venue": ""}
out = run(lambda *a, **k: [item(REFUSED)],
          lambda *a, **k: [item(None, papers=[paper])])
check("papers from the classic pass keep the run alive",
      isinstance(out, dict) and len(out["pool"]) == 1, out.get("pool"))

# 5. the audit log still records each query's error for the UI
out = run(lambda *a, **k: [item(None)], lambda *a, **k: [item("openalex: 429")])
logged = out["retrieval_log"][0]["queries"]
check("per-query errors stay in the retrieval log for the audit trail",
      [q["error"] for q in logged] == [None, "openalex: 429"], logged)

# 6. every breaker is open: no request was sent, so this is not an empty result
skip = {"source": "openalex", "hits": 0, "error": None, "skipped": True,
        "skip_reason": "breaker", "note": "近期失败，已临时跳过"}
try:
    run(lambda *a, **k: [item(None, sources=[skip])],
        lambda *a, **k: [item(None, sources=[skip])])
    check("all sources in breaker cooldown raises", False, "no exception")
except RuntimeError as e:
    msg = str(e)
    check("all sources in breaker cooldown raises", True)
    check("the breaker message is not 'no relevant papers'",
          "不是「没有相关文献」" in msg, msg)
    check("it names the skipped source", "openalex" in msg, msg)

print()
print("ALL OK" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
