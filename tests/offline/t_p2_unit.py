"""P2 bounds: deadlines, CSV, PDF inflate, caches, rate limits and owner settings."""

import os
import tempfile
import threading
import time
import zlib

import _paths  # noqa: F401
import exports
import fulltext
import retraction
import scoring
import screening
import search
import web

fails = []


def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + repr(extra)))
    if not cond:
        fails.append(name)


# ---- retraction deadline is a wall-clock cap, not an executor join ----------
old_crossref = retraction._crossref_one
try:
    def slow(doi, timeout):
        time.sleep(0.5)
        return None

    retraction._crossref_one = slow
    deadline = time.monotonic() + 0.05
    started = time.monotonic()
    retraction._run_crossref([("doi:1", "10.1/x", {"title": "T"})], {},
                             deadline, budget=1)
    elapsed = time.monotonic() - started
    check("Crossref stage returns at its deadline", elapsed < 0.25, elapsed)
finally:
    retraction._crossref_one = old_crossref


# ---- CSV formula injection ------------------------------------------------
check("a formula-looking cell is prefixed", exports._csv_cell("=HYPERLINK(1)").startswith("'="))
check("a plus formula is prefixed", exports._csv_cell("+1+1").startswith("'+"))
check("an at-command cell is prefixed", exports._csv_cell("@cmd").startswith("'@"))
check("a negative number stays numeric", exports._csv_cell(-5) == "-5")
check("a float stays numeric", exports._csv_cell(1.5) == "1.5")
check("a normal string is unchanged", exports._csv_cell("paper") == "paper")


# ---- screening path boundary ---------------------------------------------
old_screening_dir = screening.SCREENING_DIR
with tempfile.TemporaryDirectory() as td:
    screening.SCREENING_DIR = td
    try:
        check("screening rejects ..", screening._path("..") == "")
        check("screening rejects .", screening._path(".") == "")
        path = screening._path("20261021_120000_123456")
        check("screening keeps a normal id inside its directory",
              os.path.dirname(os.path.realpath(path)) == os.path.realpath(td), path)
    finally:
        screening.SCREENING_DIR = old_screening_dir


# ---- PDF inflate cap ------------------------------------------------------
old_inflate_cap = fulltext.MAX_INFLATE_BYTES
try:
    fulltext.MAX_INFLATE_BYTES = 1024
    payload = b"A" * 500
    check("a normal zlib stream still decompresses",
          fulltext._decompress(zlib.compress(payload)) == payload)
    bomb = zlib.compress(b"B" * 5000)
    check("a stream over the inflate cap is rejected", fulltext._decompress(bomb) == b"")
    raw = zlib.compressobj(wbits=-15)
    raw_stream = raw.compress(payload) + raw.flush()
    check("raw deflate still decompresses", fulltext._decompress(raw_stream) == payload)
finally:
    fulltext.MAX_INFLATE_BYTES = old_inflate_cap


# ---- bounded in-memory caches --------------------------------------------
old_entries, old_chars = fulltext.MEMO_MAX_ENTRIES, fulltext.MEMO_MAX_CHARS
try:
    fulltext._memo.clear()
    fulltext.MEMO_MAX_ENTRIES, fulltext.MEMO_MAX_CHARS = 2, 10_000
    for key in ("a", "b", "c"):
        fulltext._memo_put(key, key * 10, "")
    check("memo keeps the newest entries", list(fulltext._memo) == ["b", "c"], list(fulltext._memo))
    fulltext._memo.clear()
    fulltext.MEMO_MAX_ENTRIES, fulltext.MEMO_MAX_CHARS = 10, 6
    fulltext._memo_put("x", "1234", "")
    fulltext._memo_put("y", "5678", "")
    check("memo also enforces its character cap", list(fulltext._memo) == ["y"], list(fulltext._memo))
    value = fulltext._memo_put("huge", "z" * 20, "note")
    check("an over-cap result is returned even when it is not retained",
          value == ("z" * 20, "note") and "huge" not in fulltext._memo, value)
finally:
    fulltext._memo.clear()
    fulltext.MEMO_MAX_ENTRIES, fulltext.MEMO_MAX_CHARS = old_entries, old_chars

old_oa_cap = fulltext.OA_URL_MAX_ENTRIES
try:
    fulltext.OA_URL_MAX_ENTRIES = 2
    fulltext._oa_urls.clear()
    fulltext.set_oa_urls({"a": "https://a/1", "b": "https://b/1", "c": "https://c/1"})
    check("OpenAlex URL map keeps only the newest addresses",
          list(fulltext._oa_urls) == ["b", "c"], list(fulltext._oa_urls))
finally:
    fulltext._oa_urls.clear()
    fulltext.OA_URL_MAX_ENTRIES = old_oa_cap


# ---- search thread-local opts must be reset explicitly --------------------
search._set_request_opts({"timeout": 7, "retries": 2})
check("source opts are set", (search._REQUEST_OPTS.timeout,
                              search._REQUEST_OPTS.retries) == (7, 2))
search._set_request_opts({"timeout": None, "retries": None})
check("a source without opts clears the previous values",
      (search._REQUEST_OPTS.timeout, search._REQUEST_OPTS.retries) == (None, None))


# ---- graph endpoint rate limiter -----------------------------------------
old_rate_state = web._graph_rate
old_window, old_max = web.GRAPH_RATE_WINDOW, web.GRAPH_RATE_MAX
try:
    web._graph_rate = {}
    web.GRAPH_RATE_WINDOW, web.GRAPH_RATE_MAX = 10, 2
    check("first hit allowed", web._rate_ok("o1", "citegraph", now=100.0))
    check("second hit allowed", web._rate_ok("o1", "citegraph", now=101.0))
    check("third hit in the window rejected", not web._rate_ok("o1", "citegraph", now=102.0))
    check("a different owner is independent", web._rate_ok("o2", "citegraph", now=102.0))
    check("the window slides", web._rate_ok("o1", "citegraph", now=113.0))
finally:
    web._graph_rate = old_rate_state
    web.GRAPH_RATE_WINDOW, web.GRAPH_RATE_MAX = old_window, old_max


# ---- scoring settings are per owner, with a global fallback --------------
old_settings = scoring.SETTINGS_PATH
owner1, owner2 = "a" * 32, "b" * 32
with tempfile.TemporaryDirectory() as td:
    scoring.SETTINGS_PATH = os.path.join(td, "scoring_weights.json")
    try:
        scoring.save_settings({"relevance": 1}, owner=owner1)
        scoring.save_settings({"relevance": 5}, owner=owner2)
        scoring.save_settings({"relevance": 3})
        check("owner 1 reads its own weights",
              scoring.load_settings(owner1)["weights"]["relevance"] == 1)
        check("owner 2 reads its own weights",
              scoring.load_settings(owner2)["weights"]["relevance"] == 5)
        check("the CLI/global file is separate",
              scoring.load_settings()["weights"]["relevance"] == 3)
        check("owner files live next to the global file",
              os.path.exists(os.path.join(td, f"scoring-{owner1}.json")))
    finally:
        scoring.SETTINGS_PATH = old_settings


print()
if fails:
    print("FAILED:", ", ".join(fails))
    raise SystemExit(1)
print("all checks passed")
