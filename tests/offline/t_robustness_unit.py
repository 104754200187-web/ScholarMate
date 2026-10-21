"""Regression checks for the three robustness fixes: ZIP byte budget, deadline
skip markers, and DOI lookup failure semantics."""

import os
import tempfile
import urllib.error

import _paths  # noqa: F401
import citationgraph
import fulltext
import pdfpack
import search

fails = []


def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + repr(extra)))
    if not cond:
        fails.append(name)


# ---- 1. a PDF pack reserves the shared byte budget before downloading ------
old_fetch = pdfpack._fetch
old_total = pdfpack.MAX_TOTAL_BYTES
old_file = pdfpack.MAX_FILE_BYTES
old_workers = pdfpack.WORKERS
old_budget = pdfpack.BUDGET_S
old_oa = search.openalex_oa_pdf_urls
try:
    payload = b"%PDF-1.4" + b"x" * 4       # 12 bytes
    calls = []

    def fake_fetch(url, max_bytes=None):
        calls.append(max_bytes)
        if max_bytes is not None and len(payload) > max_bytes:
            raise RuntimeError("response too large")
        return payload

    pdfpack._fetch = fake_fetch
    pdfpack.MAX_TOTAL_BYTES = len(payload)
    pdfpack.MAX_FILE_BYTES = len(payload)
    pdfpack.WORKERS = 3
    pdfpack.BUDGET_S = 5
    papers = [{"title": f"paper {i}", "pdf_url": f"https://pdf.example/{i}.pdf"}
              for i in range(3)]
    results = pdfpack.download_papers(papers, oa={})
    kept = [r for r in results if r["ok"]]
    check("only one download starts when the whole budget is one file", len(calls) == 1, calls)
    check("only that download is retained", len(kept) == 1, results)
    check("the retained bytes fit the pack cap",
          sum(len(r["data"]) for r in kept) <= pdfpack.MAX_TOTAL_BYTES)
    check("papers rejected before download are marked oversized",
          len([r for r in results if r.get("oversized")]) == 2, results)
    check("rejected results do not retain their response body",
          all("data" not in r for r in results if not r["ok"]), results)

    search.openalex_oa_pdf_urls = lambda ps: {}
    calls.clear()
    _, summary = pdfpack.build_zip(papers)
    check("the ZIP summary reports the budget rejections",
          summary["pdfs"] == 1 and len(summary["oversized"]) == 2, summary)
finally:
    pdfpack._fetch = old_fetch
    pdfpack.MAX_TOTAL_BYTES = old_total
    pdfpack.MAX_FILE_BYTES = old_file
    pdfpack.WORKERS = old_workers
    pdfpack.BUDGET_S = old_budget
    search.openalex_oa_pdf_urls = old_oa


# ---- 2. a stage deadline is not a property of the URL ----------------------
old_cache = fulltext.FULLTEXT_CACHE
old_http = fulltext._http_get
paper = {"title": "deadline paper", "pdf_url": "https://pdf.example/deadline.pdf"}
with tempfile.TemporaryDirectory() as td:
    fulltext.FULLTEXT_CACHE = td
    fulltext._memo.clear()
    fulltext._skips.clear()
    fulltext._skips_loaded = True
    called = []
    fulltext._http_get = lambda *a, **k: called.append(a) or b"%PDF-1.4"
    try:
        fulltext.set_deadline(-1)
        text, note = fulltext.get_fulltext(paper)
        key = fulltext._cache_key(paper)
        check("a deadline returns no text", text == "" and "deadline reached" in note,
              (text, note))
        check("a deadline does not even start the download", called == [], called)
        check("a deadline writes no long-lived skip marker",
              fulltext._skip_reason(key, paper["pdf_url"]) == "",
              fulltext._skips)

        fulltext._skips.clear()
        fulltext._memo.clear()
        fulltext.set_deadline(10)

        def forbidden(*a, **k):
            raise RuntimeError("download failed: HTTP Error 403: Forbidden")

        fulltext._http_get = forbidden
        text, note = fulltext.get_fulltext(paper)
        check("a terminal failure still gets a skip marker",
              "403" in fulltext._skip_reason(key, paper["pdf_url"]),
              fulltext._skips)
    finally:
        fulltext.clear_deadline()
        fulltext._memo.clear()
        fulltext._skips.clear()
fulltext.FULLTEXT_CACHE = old_cache
fulltext._http_get = old_http


# ---- 3. DOI lookup failures are not "paper not found" ---------------------
old_get = citationgraph._get
try:
    def offline(*a, **k):
        raise citationgraph.GraphError("offline")

    citationgraph._get = offline
    failures = []
    got = citationgraph._by_dois(["10.1000/offline"], failures=failures)
    check("network failure leaves the DOI unresolved", got == {}, got)
    check("network failure is recorded for the caller", failures, failures)

    def missing(*a, **k):
        raise urllib.error.HTTPError("https://api.openalex.org/works", 404,
                                     "Not Found", None, None)

    citationgraph._get = missing
    failures = []
    got = citationgraph._by_dois(["10.1000/missing"], failures=failures)
    check("a 404 is a normal no-match, not a transport failure",
          got == {} and failures == [], (got, failures))
finally:
    citationgraph._get = old_get


old_by_dois = citationgraph._by_dois
old_handles = citationgraph._s2_handles
old_s2_refs = citationgraph._s2_refs
try:
    def failed_lookup(dois, workers=4, failures=None):
        if failures is not None:
            failures.append("offline")
        return {}

    citationgraph._by_dois = failed_lookup
    citationgraph._s2_handles = lambda papers, max_pool: [
        (0, "DOI:10.1000/a", "10.1000/a"),
        (1, "DOI:10.1000/b", "10.1000/b"),
    ]
    citationgraph._s2_refs = lambda ids, **kw: {}
    try:
        citationgraph.build_pool([
            {"title": "a", "doi": "10.1000/a"},
            {"title": "b", "doi": "10.1000/b"},
        ])
        error = None
    except citationgraph.GraphError as exc:
        error = exc
    except LookupError as exc:
        error = exc
    check("an all-failed pool lookup is a retryable GraphError, not a 404",
          isinstance(error, citationgraph.GraphError), error)
finally:
    citationgraph._by_dois = old_by_dois
    citationgraph._s2_handles = old_handles
    citationgraph._s2_refs = old_s2_refs


print()
if fails:
    print("FAILED:", ", ".join(fails))
    raise SystemExit(1)
print("all checks passed")
