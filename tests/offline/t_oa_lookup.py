"""search.openalex_oa_pdf_urls: batching, dedup, DOI normalisation, failure."""
import os, sys
import _paths  # noqa: F401
import search

fails = []
def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + repr(extra)))
    if not cond: fails.append(name)

calls = []
def stub(results_by_batch, error=None):
    def fake(url, params, timeout=None, retries=None, headers=None):
        calls.append(params)
        if error:
            raise error
        chunk = params["filter"].split("doi:", 1)[1].split("|")
        return {"results": [m for c in chunk for m in results_by_batch.get(c, [])]}
    return fake

real = search._http_get_json
def work(dois):
    return [{"doi": d, "best_oa_location": {"pdf_url": f"https://oa/{i}.pdf"}}
            for i, d in enumerate(dois)]

try:
    # two DOIs per request: five DOIs must not fit in one call
    calls.clear()
    search._http_get_json = stub({d: work([d]) for d in
                                  ["10.1/a", "10.1/b", "10.1/c", "10.1/d", "10.1/e"]})
    papers = [{"doi": f"10.1/{c}"} for c in "abcde"]
    out = search.openalex_oa_pdf_urls(papers, batch=2, timeout=1)
    check("one request per batch, not per DOI", len(calls) == 3, calls)
    check("the batches partition the DOIs",
          [len(c["filter"].split("|")) for c in calls] == [2, 2, 1],
          [c["filter"] for c in calls])
    check("every DOI comes back with its URL", sorted(out) == [f"10.1/{c}" for c in "abcde"], out)
    check("the request asks for every field a PDF can hide in",
          calls[0]["select"] ==
          "doi,best_oa_location,primary_location,open_access,locations", calls[0]["select"])

    # duplicates and blank DOIs must not be asked about twice
    calls.clear()
    search._http_get_json = stub({"10.1/a": work(["10.1/a"])})
    out = search.openalex_oa_pdf_urls(
        [{"doi": "10.1/a"}, {"doi": "10.1/A"}, {"doi": "https://doi.org/10.1/a"},
         {"doi": ""}, {"title": "no doi"}, {"doi": None}], batch=50, timeout=1)
    check("a DOI asked about in three spellings is queried once",
          calls[0]["filter"] == "doi:10.1/a", calls[0]["filter"])
    check("papers without a DOI are left out instead of sent as empty",
          len(calls) == 1 and len(out) == 1, (calls, out))

    # the DOI OpenAlex echoes back is normalised, not copied verbatim
    calls.clear()
    search._http_get_json = stub({"10.1/a": [{
        "doi": "https://doi.org/10.1/A", "best_oa_location": {"pdf_url": "https://oa/a.pdf"}}]})
    out = search.openalex_oa_pdf_urls([{"doi": "https://doi.org/10.1/a"}], timeout=1)
    check("the key is the normalised DOI, so callers can look it up",
          out == {"10.1/a": "https://oa/a.pdf"}, out)

    # best_oa_location -> primary_location -> open_access.oa_url
    loc = [("best_oa_location wins",
            {"best_oa_location": {"pdf_url": "https://oa/best.pdf"},
             "primary_location": {"pdf_url": "https://oa/prim.pdf"},
             "open_access": {"oa_url": "https://oa/oa"}}, "https://oa/best.pdf"),
           ("primary_location when there is no best",
            {"primary_location": {"pdf_url": "https://oa/prim.pdf"},
             "open_access": {"oa_url": "https://oa/oa"}}, "https://oa/prim.pdf"),
           ("a primary_location .pdf url",
            {"primary_location": {"url": "https://oa/landing.pdf"}}, "https://oa/landing.pdf"),
           ("open_access.oa_url as the last resort",
            {"open_access": {"oa_url": "https://oa/last"}}, "https://oa/last"),
           ("nothing anywhere", {"open_access": {"is_oa": False}}, ""),
           ("a landing page URL is not a PDF",
            {"primary_location": {"url": "https://oa/article"}}, "")]
    for label, workrec, want in loc:
        check("address preference: " + label, search._oa_pdf_url(workrec) == want,
              (search._oa_pdf_url(workrec), want))

    # a real file anywhere in `locations` beats a landing page
    loc = [("a repository PDF in a third location beats the doi.org resolver",
            {"open_access": {"oa_url": "https://doi.org/10.1/x"},
             "best_oa_location": {"url": "https://doi.org/10.1/x"},
             "locations": [{"url": "https://doi.org/10.1/x"},
                           {"pdf_url": "https://repo.example/x.pdf"}]},
            "https://repo.example/x.pdf"),
           ("an arXiv pre-print of a paywalled paper is found there",
            {"open_access": {"oa_url": "https://doi.org/10.1145/1"},
             "primary_location": {"url": "https://doi.org/10.1145/1"},
             "locations": [{"url": "https://doi.org/10.1145/1"},
                           {"pdf_url": "https://arxiv.org/pdf/2401.14887"}]},
            "https://arxiv.org/pdf/2401.14887"),
           ("a location that only links its landing page is not used",
            {"open_access": {"oa_url": "https://doi.org/10.1/x"},
             "locations": [{"url": "https://repo.example/x"}, {"pdf_url": ""}]},
            "https://doi.org/10.1/x"),
           ("the same file listed twice is not returned twice",
            {"locations": [{"url": "https://a/b.pdf"}, {"pdf_url": "https://a/b.pdf"}]},
            "https://a/b.pdf")]
    for label, workrec, want in loc:
        check("file before landing page: " + label,
              search._oa_pdf_url(workrec) == want, (search._oa_pdf_url(workrec), want))

    # a work with no usable URL is simply absent, and other DOIs still resolve
    calls.clear()
    search._http_get_json = stub({"10.1/a": [{"doi": "10.1/a", "open_access": {}}],
                                  "10.1/b": work(["10.1/b"])})
    out = search.openalex_oa_pdf_urls([{"doi": "10.1/a"}, {"doi": "10.1/b"}], batch=50, timeout=1)
    check("a work with no open copy is skipped, the rest still resolve",
          list(out) == ["10.1/b"], out)

    # never fatal: a failing request is not an exception
    calls.clear()
    search._http_get_json = stub({}, error=RuntimeError("HTTP Error 502"))
    try:
        out = search.openalex_oa_pdf_urls([{"doi": "10.1/a"}], timeout=1)
        check("a failed lookup returns {} instead of raising", out == {}, out)
    except Exception as e:
        check("a failed lookup returns {} instead of raising", False, e)

    # ...and one bad batch does not discard the good ones
    calls.clear()
    state = {"n": 0}
    def flaky(url, params, timeout=None, retries=None, headers=None):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("HTTP Error 502")
        return {"results": work(["10.1/b"])}
    search._http_get_json = flaky
    out = search.openalex_oa_pdf_urls([{"doi": "10.1/a"}, {"doi": "10.1/b"}], batch=1, timeout=1)
    check("one failing batch does not lose the others", list(out) == ["10.1/b"], out)

    calls.clear()
    search._http_get_json = stub({})
    check("an empty list asks nothing and returns {}",
          search.openalex_oa_pdf_urls([], timeout=1) == {} and calls == [])
finally:
    search._http_get_json = real

print()
print("ALL OK" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
