"""Literature retrieval layer, pure stdlib.

Sources: Semantic Scholar + OpenAlex + Google Scholar (SerpAPI), plus a
coverage-oriented second tier of arXiv, PubMed/Europe PMC and Crossref.

`search(..., source="auto")` queries every available source **concurrently**
and merges the results (deduped by normalized title, interleaved by rank) to
widen coverage; if one source fails the others still return. Sources live in a
registry (`register_source`) with per-source throttles, so adding another
database later needs no changes anywhere else. Set S2_API_KEY in .env for a
generous Semantic Scholar rate limit; OPENALEX_MAILTO joins the OpenAlex
polite-pool limit.

Google Scholar has NO official API and scholar.google.com blocks direct
scraping, so Scholar results are only available through a paid third-party
proxy (SerpAPI, https://serpapi.com). When SERPAPI_KEY is set in .env,
source="auto" also queries Google Scholar via SerpAPI and merges its hits; the
source stays S2 + OpenAlex when the key is absent.

Besides the relevance pass there is a *citation-ranked* companion pass
(``search_classics``): it re-asks the three sources that can sort by citation
count (Semantic Scholar bulk search, OpenAlex, Crossref) for the most cited
works matching the same query, so a review does not silently consist of
whatever is merely new.
"""

import contextvars
import contextlib
import email.utils
import inspect
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree
from concurrent.futures import ThreadPoolExecutor, as_completed

import fulltext
import journal

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# The relevance endpoint cannot sort; the bulk endpoint searches the whole
# corpus and can, which is what the citation-ranked pass needs.
S2_BULK_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "title,abstract,year,citationCount,url,authors,externalIds,openAccessPdf,venue"
ARXIV_URL = "http://export.arxiv.org/api/query"
EUROPEPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
OPENALEX_SEARCH_URL = "https://api.openalex.org/works"
# Europe PMC answers a *date-ordered* page and has no relevance sort (see
# search_europe_pmc), so ask for a wider page and rank it here.
EUROPEPMC_PAGE = int(os.environ.get("EUROPEPMC_PAGE", "50"))
SERPAPI_SEARCH_URL = "https://serpapi.com/search"


def openalex_auth(params):
    """Attach the OpenAlex credential (API key and/or polite-pool mailto)."""
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO
    return params


def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()
S2_API_KEY = os.environ.get("S2_API_KEY")
OPENALEX_MAILTO = os.environ.get("OPENALEX_MAILTO")
# OpenAlex sheds load by throttling anonymous traffic first ("Anonymous search
# is temporarily rate-limited while the search cluster is under elevated load");
# the polite-pool mailto does not lift that one, a free API key does.
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

# How many (query x source) requests may be in flight at once. Sources are hit
# in parallel; per-source throttles below keep the politeness rules.
SEARCH_WORKERS = int(os.environ.get("SEARCH_WORKERS", "6"))

# Citation-ranked ("classic anchor") companion pass: how many hits to ask per
# query, and how many citations a paper needs to count as a landmark. A hit
# list with no landmark at all still contributes its single most cited paper,
# so a niche topic gets its local anchor instead of nothing.
CLASSIC_PER_QUERY = int(os.environ.get("CLASSIC_PER_QUERY", "8"))
CLASSIC_MIN_CITATIONS = int(os.environ.get("CLASSIC_MIN_CITATIONS", "100"))


def _title_key(title):
    """Normalized title used to dedupe papers across sources."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


_REQUEST_OPTS = threading.local()

# A 429 that comes with a long Retry-After is not a hiccup, it is a ban: back
# off for as long as the server asks (capped) instead of burning the retry
# budget in every round of every run.
RETRY_AFTER_GIVEUP_S = 60
MAX_COOLDOWN_S = float(os.environ.get("SOURCE_MAX_COOLDOWN_S", "1800"))


class RateLimited(RuntimeError):
    """Raised when a source asks us to come back much later."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _parse_retry_after(value):
    """Seconds from a Retry-After header (delta-seconds or HTTP date)."""
    if not value:
        return None
    value = str(value).strip()
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


def _http_get_text(url, params, timeout=None, retries=None, headers=None):
    # a source may declare its own timeout/retries (see register_source); the
    # thread-local carries that choice into the shared request helper
    timeout = getattr(_REQUEST_OPTS, "timeout", None) or timeout or 30
    retries = getattr(_REQUEST_OPTS, "retries", None) or retries or 4
    # %20 rather than "+" for spaces: arXiv's legacy API answers 406 to "+"
    query = urllib.parse.urlencode(params or {}, quote_via=urllib.parse.quote)
    req_headers = {"User-Agent": "research-agent/0.1"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(f"{url}?{query}" if query else url, headers=req_headers)
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 503):  # rate limited: back off with jitter and retry
                retry_after = _parse_retry_after(e.headers.get("Retry-After"))
                if retry_after and retry_after > RETRY_AFTER_GIVEUP_S:
                    raise RateLimited(
                        f"HTTP {e.code}: server asked us to wait {retry_after:.0f}s",
                        retry_after) from None
                time.sleep(min(2 ** attempt, 16) + (time.monotonic() % 1))
                continue
            raise
        except urllib.error.URLError as e:
            last_err = e
            reason = str(getattr(e, "reason", e))
            # A socket that timed out or failed the TLS handshake does not come
            # back a second later: in a parallel query set every source waits
            # for the slowest one, so retrying here just doubles that wait
            # (measured: Europe PMC / Crossref paid 43s per query this way).
            if _is_connection_failure(reason):
                raise RuntimeError(f"connection failed: {reason}") from None
            time.sleep(1 + attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {last_err}")


def _http_get_json(url, params, timeout=None, retries=None, headers=None):
    text = _http_get_text(url, params, timeout=timeout, retries=retries, headers=headers)
    try:
        return json.loads(text)
    except ValueError as e:
        raise RuntimeError(f"non-JSON response from {url}: {e}") from None


def http_get_json(url, params=None, **kwargs):
    """Public wrapper around the request helper, for modules that talk to APIs
    which are not registered search sources (see retraction.py)."""
    return _http_get_json(url, params or {}, **kwargs)


def _normalize_paper(
    title, abstract, year, citations, url, authors, source,
    pdf_url="", arxiv_id="", doi="", venue="",
):
    return {
        "title": title or "",
        "abstract": abstract or "",
        "year": year,
        "citations": citations if citations is not None else 0,
        "url": url or "",
        "authors": authors or [],
        "source": source,
        "pdf_url": pdf_url or "",
        "arxiv_id": arxiv_id or "",
        "doi": doi or _derive_doi(url),
        "venue": venue or "",
        "seen_sources": [source],
    }


def _derive_doi(url):
    """Best-effort DOI extraction from a URL (doi.org/10.xxxx/...)."""
    if not url:
        return ""
    m = re.search(r"(?:doi\.org/|doi=|doi:)\s*([^?#\s]+)", url)
    return m.group(1) if m else ""


def _apply_filters(papers, filters):
    """Client-side post filter for all sources (semantics kept consistent).

    - year window: drop papers whose year is missing or outside [lo, hi]
    - review_only: keep titles matching "survey" / "review"
    - venue: keep papers whose journal/conference name contains this text
    - venue class (正刊/子刊/一区-四区): the run's own journal.TierFilter, read
      from the context variable the graph sets for the run -- see journal.py for
      where a class comes from. Applying it here is what makes every source
      filter identically; the cheap tests run first so the tally the UI shows
      only counts papers that got past the other filters.
    """
    f = filters or {}
    lo = f.get("min_year")
    hi = f.get("max_year")
    review_only = bool(f.get("review_only"))
    venue_q = str(f.get("venue") or "").strip()
    tier = journal.current_filter()
    tier_active = bool(tier is not None and tier.enabled)
    if not (lo or hi or review_only or venue_q or tier_active):
        return papers
    out = []
    for p in papers:
        year = p.get("year")
        if (lo or hi) and (year is None):
            continue
        if lo and year < lo:
            continue
        if hi and year > hi:
            continue
        if review_only and not re.search(r"\b(review|survey)\b", p.get("title", ""), re.I):
            continue
        if venue_q and venue_q.lower() not in (p.get("venue") or "").lower():
            continue
        out.append(p)
    if tier_active:
        out = tier.filter(out)
    return out


class _Throttle:
    """Per-source politeness: cap concurrency and enforce a minimum gap."""

    def __init__(self, max_concurrency=2, min_gap=0.0):
        self._sem = threading.Semaphore(max(1, max_concurrency))
        self._lock = threading.Lock()
        self._min_gap = max(0.0, min_gap)
        self._last = 0.0

    @contextlib.contextmanager
    def slot(self):
        self._sem.acquire()
        try:
            if self._min_gap:
                with self._lock:
                    gap = self._min_gap - (time.monotonic() - self._last)
                    if gap > 0:
                        time.sleep(gap)
                    self._last = time.monotonic()
            yield
        finally:
            self._sem.release()


_SOURCES = []      # ordered registry of {"name", "label", "fn", "available"}
_THROTTLES = {}    # name -> _Throttle
_BREAKER = {}      # name -> monotonic deadline; a tripped source is skipped
# A source that just failed is very likely to fail again (rate limits, a
# blocking CDN, an outage). Skipping it for a while keeps one dead database from
# paying its retry/backoff budget in every round of every run; adding coverage
# must never mean adding wall clock.
SOURCE_COOLDOWN_S = float(os.environ.get("SOURCE_COOLDOWN_S", "180"))


_FAILS = {}        # name -> consecutive failures (drives the cool-down)


def _is_connection_failure(reason):
    """True for errors where a retry is pointless (timeouts, TLS, resets)."""
    reason = reason.lower()
    return any(k in reason for k in
               ("timed out", "timeout", "handshake", "connection reset",
                "connection refused", "unreachable"))


def _breaker_open(name):
    return time.monotonic() < _BREAKER.get(name, 0.0)


def _trip_breaker(name, seconds=None):
    """Park a failed source. `seconds` (e.g. a Retry-After) only ever extends
    the cool-down; it is capped so we still probe for recovery periodically.
    A source that keeps failing (a blocked CDN, an outage) is parked for twice
    as long each time, instead of being probed every SOURCE_COOLDOWN_S."""
    fails = _FAILS.get(name, 0) + 1
    _FAILS[name] = fails
    wait = max(float(seconds or 0), SOURCE_COOLDOWN_S * (2 ** (fails - 1)))
    _BREAKER[name] = time.monotonic() + min(wait, MAX_COOLDOWN_S)


def _clear_breaker(name):
    _BREAKER.pop(name, None)
    _FAILS.pop(name, None)


def breaker_status():
    """Sources currently parked after a failure, for the UI/log."""
    return {n: round(d - time.monotonic(), 1) for n, d in _BREAKER.items()
            if d > time.monotonic()}


def breaker_open(name):
    """True while a source is parked; other modules (retraction.py) use this to
    skip a database they already know is unreachable."""
    return _breaker_open(name)


def register_source(name, fn, label=None, available=True, max_concurrency=2, min_gap=0.0,
                    timeout=None, retries=None):
    """Register (or replace) a retrieval source.

    To add another database later: write ``def search_foo(query, limit)`` that
    returns papers built with ``_normalize_paper(..., source="foo")``, then call
    ``register_source("foo", search_foo, "Foo")`` at import time. It joins
    ``source="auto"`` merges automatically, gets its own throttle, and shows up
    in ``source_status()`` (surfaced in the web UI).

    Tuning knobs per source:
      max_concurrency -- how many of its requests may overlap
      min_gap         -- minimum seconds between two request *starts*
      timeout/retries -- per-request HTTP timeout and retry budget, so one
                         throttled database cannot dominate the wall clock
    """
    try:
        supports_sort = "sort" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        supports_sort = False
    spec = {"name": name, "fn": fn, "label": label or name, "available": bool(available),
            "timeout": timeout, "retries": retries, "supports_sort": supports_sort}
    for i, existing in enumerate(_SOURCES):
        if existing["name"] == name:
            _SOURCES[i] = spec
            break
    else:
        _SOURCES.append(spec)
    _THROTTLES[name] = _Throttle(max_concurrency, min_gap)
    return spec


def available_sources():
    """Names of the sources that can be queried right now."""
    return [s["name"] for s in _SOURCES if s["available"]]


def source_status():
    """Registry snapshot for the UI: every source and whether it is active."""
    return [
        {"name": s["name"], "label": s["label"], "available": s["available"]}
        for s in _SOURCES
    ]


def _enabled_sources(names=None):
    """Registered sources that are available; `names` restricts the set (used
    by the per-sentence evidence search, which only wants the fast two)."""
    return [s for s in _SOURCES
            if s["available"] and (not names or s["name"] in names)]


def _source_spec(name):
    for s in _SOURCES:
        if s["name"] == name:
            return s
    return None


def _source_fn(name):
    spec = _source_spec(name)
    return spec["fn"] if spec else None


def _call_source(spec, query, limit, sort=None, filters=None):
    """Invoke one source, passing `sort` only to sources that declare it.

    Sources without a sort parameter (arXiv has no citation counts, SerpAPI
    exposes none) simply keep answering the relevance query, so a sort request
    never turns into a broken request.
    """
    if sort and spec.get("supports_sort"):
        hits = spec["fn"](query, limit, sort=sort)
    else:
        hits = spec["fn"](query, limit)
    return _apply_filters(hits, filters) if filters else hits


_HINTS_SHOWN = set()


def _maybe_hint(name, error):
    """One-off operational hints for the source that just failed."""
    if isinstance(error, RateLimited) and name != "openalex":
            _HINTS_SHOWN.add("rate_limited")
            print(f"[hint] {name} is cooling down for "
                  f"{getattr(error, 'retry_after', 0):.0f}s (server-requested); "
                  f"it is skipped until then")
    if name == "openalex" and "429" in str(error) and not OPENALEX_API_KEY:
        if "openalex_key" not in _HINTS_SHOWN:
            _HINTS_SHOWN.add("openalex_key")
            print("[hint] OpenAlex is rate limiting us. The polite pool "
                  "(OPENALEX_MAILTO) does not lift a 429 that says \"search "
                  "cluster is under elevated load\" -- a free API key does: "
                  "https://openalex.org/ (set OPENALEX_API_KEY in .env)")
    if name == "arxiv" and "406" in str(error):
        if "arxiv_406" not in _HINTS_SHOWN:
            _HINTS_SHOWN.add("arxiv_406")
            print("[hint] arXiv answers 406 through its CDN for this network (only "
                  "already-cached URLs come back); the other sources keep working "
                  "and arXiv results still arrive via Semantic Scholar/Crossref")


def _run_source(spec, query, limit, filters, sort=None):
    """Query one source; never raises, always returns a result record."""
    name = spec["name"]
    if _breaker_open(name):
        return {"source": name, "hits": [], "error": None, "skipped": True,
                "note": "近期失败，已临时跳过", "seconds": 0.0}
    if sort and not spec.get("supports_sort"):
        return {"source": name, "hits": [], "error": None, "skipped": True,
                "note": "该源不支持按被引量排序", "seconds": 0.0}
    t0 = time.monotonic()
    try:
        with _THROTTLES[name].slot():
            _REQUEST_OPTS.timeout = spec.get("timeout")
            _REQUEST_OPTS.retries = spec.get("retries")
            hits = _call_source(spec, query, limit, sort=sort, filters=filters)
        _clear_breaker(name)
        return {"source": name, "hits": hits, "error": None,
                "seconds": round(time.monotonic() - t0, 2)}
    except Exception as e:
        if sort:
            # the citation-ranked pass is optional extra coverage: report the
            # failure but do not park a source that answers the relevance pass
            return {"source": name, "hits": [], "error": str(e),
                    "seconds": round(time.monotonic() - t0, 2)}
        _trip_breaker(name, getattr(e, "retry_after", None))
        _maybe_hint(name, e)
        return {"source": name, "hits": [], "error": str(e),
                "seconds": round(time.monotonic() - t0, 2)}


def _query_sources_many(queries, limit, filters=None, sort=None, sources=None):
    """Run every enabled source for every query, all concurrently.

    Returns {query_index: {"results": [...], "errors": [...]}} with the source
    order preserved no matter which thread finished first, so merging stays
    deterministic. Warnings are printed once, after all tasks settle.
    `sources` (an iterable of names) restricts the run to those sources.
    """
    specs = [s for s in _enabled_sources(sources) if not sort or s.get("supports_sort")]
    if not specs:
        return {i: {"results": [], "errors": ["no search sources enabled"]}
                for i in range(len(queries))}
    out = {qi: {"results": [None] * len(specs), "errors": []} for qi in range(len(queries))}
    tasks = [(qi, si, spec)
             for qi, _ in enumerate(queries)
             for si, spec in enumerate(specs)]
    workers = max(1, min(SEARCH_WORKERS, len(tasks)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        # a context copy per task: the worker's console lines belong to the run
        # that started it, and it has to see that run's stop request
        futures = {
            ex.submit(contextvars.copy_context().run, _run_source,
                      spec, queries[qi], limit, filters, sort): (qi, si)
            for qi, si, spec in tasks
        }
        for fut in as_completed(futures):
            qi, si = futures[fut]
            out[qi]["results"][si] = fut.result()
    for qi in sorted(out):
        for r in out[qi]["results"]:
            if r and r["error"]:
                out[qi]["errors"].append(f"{r['source']}: {r['error']}")
                print(f"[warn] {r['source']} search failed ({r['error']}); continuing with the other source")
    return out


def _query_sources(query, limit, filters=None):
    """Single-query wrapper around the parallel runner."""
    slot = _query_sources_many([query], limit, filters)[0]
    return slot["results"], slot["errors"]


def _mark_seen(paper, source):
    """Return a paper whose provenance list includes `source` (shallow copy)."""
    seen = paper.get("seen_sources") or [paper.get("source")]
    if source not in seen:
        seen = seen + [source]
    if seen == paper.get("seen_sources"):
        return paper
    out = dict(paper)
    out["seen_sources"] = seen
    return out


def _merge_hits(results):
    """Interleave hits by per-source rank; dedupe by normalized title while
    keeping the full provenance (seen_sources) and preferring abstracts."""
    merged = {}
    for i in range(max((len(r["hits"]) for r in results), default=0)):
        for r in results:
            if i >= len(r["hits"]):
                continue
            p = r["hits"][i]
            key = _title_key(p["title"])
            if not key:
                continue
            prev = merged.get(key)
            if prev is None:
                merged[key] = p
            elif not prev.get("abstract") and p.get("abstract"):
                merged[key] = _mark_seen(p, prev["source"])
            else:
                merged[key] = _mark_seen(prev, r["source"])
    return list(merged.values())


# ------------------------------------------------------- second-tier sources
# Added for *coverage* rather than for a specific discipline. All three are free
# and key-less; each is a plain function returning _normalize_paper dicts, wired
# in through register_source() at the bottom of this module.

def search_arxiv(query, limit=20):
    """arXiv API (Atom XML): preprints in CS/physics/math/quant-bio.

    arXiv carries no citation counts and no venue, but it is the primary record
    for CS/ML work long before it appears anywhere else, and it always has an
    abstract plus a PDF link the full-text evidence stage can use.
    """
    cleaned = re.sub(r'[":?*]', " ", query).strip()
    if not cleaned:
        return []
    params = {
        "search_query": f"all:{cleaned}",
        "start": 0,
        "max_results": max(1, min(limit, 50)),
        "sortBy": "relevance",
    }
    text = _http_get_text(ARXIV_URL, params)
    try:
        root = xml.etree.ElementTree.fromstring(text)
    except xml.etree.ElementTree.ParseError as e:
        raise RuntimeError(f"arXiv returned unparseable XML: {e}") from None
    papers = []
    for entry in root.findall("atom:entry", _ATOM_NS):
        title = _clean_text(entry.findtext("atom:title", "", _ATOM_NS))
        if not title:
            continue
        abstract = _clean_text(entry.findtext("atom:summary", "", _ATOM_NS))
        raw_id = entry.findtext("atom:id", "", _ATOM_NS)
        arxiv_id = re.sub(r"^https?://arxiv\.org/abs/", "", raw_id or "").strip()
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        pdf_url = ""
        for link in entry.findall("atom:link", _ATOM_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href") or ""
        published = entry.findtext("atom:published", "", _ATOM_NS) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            _clean_text(a.findtext("atom:name", "", _ATOM_NS))
            for a in entry.findall("atom:author", _ATOM_NS)
        ]
        cat = entry.find("arxiv:primary_category", _ATOM_NS)
        papers.append(_normalize_paper(
            title=title,
            abstract=abstract,
            year=year,
            citations=None,
            url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else raw_id,
            authors=[a for a in authors if a],
            source="arxiv",
            pdf_url=pdf_url,
            arxiv_id=arxiv_id,
            doi=f"10.48550/arXiv.{arxiv_id}" if arxiv_id else "",
            venue=(cat.get("term") if cat is not None else "") or "",
        ))
    return papers


def _relevance_order(query, papers, limit):
    """Order a date-sorted page by relevance to the query (local BM25).

    Titles are counted twice: a query term in the title says more than one
    buried in an abstract. Without this, the 8 slots this pipeline keeps per
    query would be "the 8 newest matches", which is what a naive page returns
    -- often papers that merely share a word with the query.
    """
    if len(papers) <= limit:
        return papers[:limit]
    docs = [fulltext.tokens(p.get("title") or "") * 2
            + fulltext.tokens(p.get("abstract") or "")
            for p in papers]
    scores = fulltext.bm25_scores(query, docs)
    order = sorted(range(len(papers)), key=lambda i: -scores[i])
    return [papers[i] for i in order[:limit]]


def search_europe_pmc(query, limit=20, page=None):
    """Europe PMC: PubMed/MEDLINE plus PMC and the life-science preprints
    (bioRxiv / medRxiv / Preprints.org) behind one key-less REST API.

    The REST API has no relevance sort: measured 2026-10-04, the default
    ordering is newest-first (a "minimum wage employment effects" query answered
    2026 clinical papers), `sort=relevance` is a 404, `sort=score desc` behaves
    like the default, and `sort=CITED desc` works but returns famous papers that
    are barely related to the query. So the relevance ordering is done on our
    side: fetch a wider page (`EUROPEPMC_PAGE`), rank it with `_relevance_order`
    and keep `limit`. The candidate *pool* is still the newest matches -- this
    source is the pipeline's recency channel, next to search_classics().
    """
    page = EUROPEPMC_PAGE if page is None else int(page)
    params = {
        "query": query,
        "format": "json",
        "pageSize": max(1, min(max(page, limit), 100)),
        "resultType": "core",
    }
    if OPENALEX_MAILTO:
        params["email"] = OPENALEX_MAILTO
    data = _http_get_json(EUROPEPMC_URL, params)
    papers = []
    for r in ((data.get("resultList") or {}).get("result") or []):
        title = _clean_text(r.get("title"))
        if not title:
            continue
        try:
            year = int(r["pubYear"]) if r.get("pubYear") else None
        except (TypeError, ValueError):
            year = None
        doi = r.get("doi") or ""
        url = f"https://doi.org/{doi}" if doi else (
            f"https://europepmc.org/article/{r.get('source') or 'MED'}/{r.get('id') or ''}"
        )
        pdf_url = ""
        if str(r.get("isOpenAccess") or "").upper() == "Y":
            for ft in ((r.get("fullTextUrlList") or {}).get("fullTextUrl") or []):
                if str(ft.get("documentStyle") or "").lower() == "pdf":
                    pdf_url = ft.get("url") or ""
                    break
        papers.append(_normalize_paper(
            title=title,
            abstract=_clean_text(r.get("abstractText")),
            year=year,
            citations=r.get("citedByCount"),
            url=url,
            authors=[a.strip() for a in (r.get("authorString") or "").split(",") if a.strip()],
            source="europe_pmc",
            pdf_url=pdf_url,
            doi=doi,
            venue=_clean_text(r.get("journalTitle")),
        ))
    return _relevance_order(query, papers, limit)


def search_crossref(query, limit=20):
    """Crossref: the registration agency for essentially every DOI, hence the
    broadest net for journal articles, book chapters, proceedings and
    non-English venues the other indexes miss.

    Deliberately has no `sort` parameter: Crossref applies `sort` before the
    query, so ?query.bibliographic=X&sort=is-referenced-by-count answers with
    the most cited works in the whole corpus (measured: "Projector augmented-
    wave method", 90k citations, for a RAG question) instead of the most cited
    works matching X. Relevance is the only usable order here.
    """
    params = {
        "query.bibliographic": query,
        "rows": max(1, min(limit, 100)),
        "select": "DOI,title,abstract,issued,is-referenced-by-count,author,"
                  "container-title,URL,type",
    }
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO
    data = _http_get_json(CROSSREF_WORKS_URL, params)
    papers = []
    for w in ((data.get("message") or {}).get("items") or []):
        titles = w.get("title") or []
        title = _clean_text(titles[0] if titles else "")
        if not title:
            continue
        parts = ((w.get("issued") or {}).get("date-parts") or [[]])[0]
        year = int(parts[0]) if parts and str(parts[0]).isdigit() else None
        doi = w.get("DOI") or ""
        authors = [
            " ".join(x for x in (a.get("given"), a.get("family")) if x)
            for a in (w.get("author") or [])
        ]
        containers = w.get("container-title") or []
        papers.append(_normalize_paper(
            title=title,
            abstract=_clean_text(w.get("abstract")),
            year=year,
            citations=w.get("is-referenced-by-count"),
            url=w.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
            authors=[a for a in authors if a],
            source="crossref",
            pdf_url="",
            doi=doi,
            venue=_clean_text(containers[0] if containers else "") or (w.get("type") or ""),
        ))
    return papers


def search_semantic_scholar(query, limit=20, sort=None):
    """Semantic Scholar. `sort="citations"` switches to the bulk endpoint, which
    searches the whole corpus in citation order instead of a relevance
    shortlist (the relevance endpoint has no sort parameter at all)."""
    headers = None
    if S2_API_KEY:
        headers = {"x-api-key": S2_API_KEY}
    if sort == "citations":
        data = _http_get_json(
            S2_BULK_SEARCH_URL,
            {"query": query, "fields": S2_FIELDS, "sort": "citationCount:desc"},
            headers=headers,
        )
    else:
        data = _http_get_json(
            S2_SEARCH_URL,
            {"query": query, "limit": limit, "fields": S2_FIELDS},
            headers=headers,
        )
    papers = []
    for p in (data.get("data") or [])[:limit]:
        ext = p.get("externalIds") or {}
        oa = p.get("openAccessPdf") or {}
        papers.append(
            _normalize_paper(
                title=p.get("title"),
                abstract=p.get("abstract"),
                year=p.get("year"),
                citations=p.get("citationCount"),
                url=p.get("url"),
                authors=[a.get("name", "") for a in p.get("authors", [])],
                source="semantic_scholar",
                pdf_url=oa.get("url", ""),
                arxiv_id=ext.get("ArXiv", ""),
                doi=ext.get("DOI", ""),
                venue=p.get("venue"),
            )
        )
    return papers


def search_many(queries, limit=5, source="auto", filters=None, sources=None):
    """Retrieve papers for several queries. Queries (and their sources) run
    concurrently, so this is bounded by the slowest single request rather than
    their sum. `sources` restricts which databases are asked (default: all)."""
    queries = [q for q in queries if q.strip()]
    if not queries:
        return {}
    return {
        d["query"]: d["papers"]
        for d in search_many_detailed(queries, limit=limit, source=source,
                                      filters=filters, sources=sources)
    }


def search_many_detailed(queries, limit=5, source="auto", filters=None, sort=None,
                         sources=None):
    """Like search_many but returns per-query provenance stats.

    Each item: {"query", "papers", "sources": [{"source", "hits", "seconds",
    "error"}], "error": str|None}. Used by retrieval snapshots/audit logs.
    `sort="citations"` asks the sources that support it to rank by citation
    count; the others are skipped rather than queried a second time.
    """
    queries = [q for q in queries if q.strip()]
    if not queries:
        return []
    if source != "auto":
        spec = _source_spec(source)
        if spec is None:
            raise ValueError(f"unknown source: {source}")
        out = []
        for q in queries:
            t0 = time.monotonic()
            item = {"query": q, "papers": [], "sources": [], "error": None}
            try:
                papers = _call_source(spec, q, limit, sort=sort, filters=filters)
                item["papers"] = papers
                item["sources"].append({
                    "source": source, "hits": len(papers),
                    "seconds": round(time.monotonic() - t0, 2), "error": None,
                })
            except Exception as e:
                item["error"] = str(e)
            out.append(item)
        return out

    parallel = _query_sources_many(queries, limit, filters, sort=sort, sources=sources)
    out = []
    for qi, q in enumerate(queries):
        slot = parallel[qi]
        out.append({
            "query": q,
            "papers": _merge_hits(slot["results"]),
            "sources": [
                {"source": r["source"], "hits": len(r["hits"]),
                 "seconds": r["seconds"], "error": r["error"],
                 "skipped": bool(r.get("skipped")), "note": r.get("note")}
                for r in slot["results"]
            ],
            "error": "; ".join(slot["errors"]) or None,
        })
    return out


_MARKUP_RE = re.compile(r"<[^>]+>")


def _clean_text(text):
    """Collapse whitespace and strip the JATS/HTML markup publishers embed in
    titles and abstracts, so downstream matching works on plain text."""
    if not text:
        return ""
    text = _MARKUP_RE.sub(" ", str(text))
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", text).strip()


def _reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[i] for i in sorted(positions))


def reconstruct_abstract(inverted_index):
    """Public wrapper (see http_get_json): citationgraph.py renders the same
    OpenAlex abstracts when a reader clicks a circle in the graph."""
    return _reconstruct_abstract(inverted_index)


def search_openalex(query, limit=20, sort=None):
    # OpenAlex treats ? and * as search wildcards, which breaks natural-language
    # questions ending in "?" with HTTP 400. Strip them before querying.
    query = query.replace("?", "").replace("*", "")
    params = {
        "search": query,
        "per-page": limit,
        "select": "title,publication_year,cited_by_count,doi,authorships,abstract_inverted_index,best_oa_location,primary_location,is_retracted",
    }
    if sort == "citations":
        # Plain `search` matches full text loosely (~35k hits for a RAG
        # question), and sorting that by citations answers with generic
        # blockbusters ("GPT-4 Technical Report"). Restricting to title +
        # abstract first makes the citation order mean "most cited work about
        # this", which is the point of the pass.
        params.pop("search", None)
        params["filter"] = "title_and_abstract.search:" + re.sub(r"[,:]", " ", query)
        params["sort"] = "cited_by_count:desc"
    openalex_auth(params)   # api_key lifts the load-shedding 429; mailto is the polite pool
    data = _http_get_json(
        OPENALEX_SEARCH_URL,
        params,
    )
    papers = []
    for w in data.get("results", []):
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in w.get("authorships", [])
        ]
        loc = w.get("best_oa_location") or {}
        source = (w.get("primary_location") or {}).get("source") or {}
        venue = source.get("display_name") or (loc.get("source") or {}).get("display_name")
        pdf_url = loc.get("pdf_url") or ""
        if not pdf_url:
            loc_url = loc.get("url") or ""
            if loc_url.lower().endswith(".pdf"):
                pdf_url = loc_url
        paper = _normalize_paper(
            title=w.get("title"),
            abstract=_reconstruct_abstract(w.get("abstract_inverted_index")),
            year=w.get("publication_year"),
            citations=w.get("cited_by_count"),
            url=w.get("doi"),
            authors=authors,
            source="openalex",
            pdf_url=pdf_url,
            venue=venue,
        )
        # OpenAlex ships Retraction Watch flags for free; retraction.py fills
        # this in for the sources that do not.
        paper["retracted"] = bool(w.get("is_retracted"))
        if paper["retracted"]:
            paper["retraction_source"] = "openalex"
            paper["retraction_note"] = "OpenAlex / Retraction Watch 标记为已撤稿"
        papers.append(paper)
    return papers


def _doi_key(value):
    """Normalise a DOI (`10.1234/x`), from a paper dict or a raw string."""
    doi = value.get("doi") if isinstance(value, dict) else value
    doi = (doi or "").strip().lower()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)


def _oa_pdf_url(work):
    """Best openly available PDF for one OpenAlex work, or "".

    A location's `pdf_url` is a file; `open_access.oa_url` is very often only a
    landing page -- of 273 papers OpenAlex could answer for, 189 (69%) came back
    as a bare doi.org resolver, i.e. an address where a bot is then refused. So
    every location is scanned for a file before any landing page is considered.
    That is also how a paywalled article's arXiv or repository copy is found:
    measured on the 412 records OpenAlex returned for the 416 cited papers that
    carry no PDF URL of their own, a file was chosen for 183 of them instead of
    152 -- 31 more papers handed the fetcher something downloadable, none
    handed it less (e.g. 10.1145/3626772.3657834: a doi.org resolver becomes
    arxiv.org/pdf/2401.14887). A DOI resolver is still better than nothing --
    Springer's landing page carries `<meta name="citation_pdf_url">`, which the
    full-text fetcher follows -- so it stays as the last resort.
    """
    def file_of(loc):
        if not isinstance(loc, dict):
            return ""
        url = (loc.get("pdf_url") or "").strip()
        if not url and (loc.get("url") or "").lower().endswith(".pdf"):
            url = loc["url"].strip()
        return url

    cands = [file_of(loc) for loc in (work.get("locations") or [])]
    cands += [file_of(work.get("best_oa_location")),
              file_of(work.get("primary_location"))]
    cands.append(((work.get("open_access") or {}).get("oa_url") or "").strip())
    cands = [u for u in dict.fromkeys(cands) if u]
    files = [u for u in cands if "doi.org/" not in u]
    return (files or cands or [""])[0]


def openalex_oa_pdf_urls(papers, batch=None, timeout=None):
    """{normalised DOI: open-access PDF URL} for the given papers. Never raises.

    One request per `batch` DOIs (OpenAlex takes 50 per filter). Two callers
    need this: the per-sentence stage, because papers that arrive with no PDF
    URL of their own are otherwise never even attempted (on the ten saved
    surveys: 84 of 243 cited papers, 35%) and a publisher answering 403 to a
    bot is the most common full-text failure on this host (50 of 102 skip
    markers); and pdfpack, for the same URLs when it packs a run's PDFs. Only
    DOIs are resolvable this way, so papers without one are simply absent from
    the result and keep their original candidate URLs.
    """
    keys = [k for k in dict.fromkeys(_doi_key(p) for p in papers) if k]
    out = {}
    batch = batch or int(os.environ.get("OPENALEX_OA_BATCH", "50"))
    timeout = timeout or int(os.environ.get("OPENALEX_OA_TIMEOUT", "15"))
    for i in range(0, len(keys), batch):
        chunk = keys[i:i + batch]
        params = {
            "filter": "doi:" + "|".join(chunk),
            "per-page": len(chunk),
            # `locations` is where a repository or arXiv copy shows up when
            # the two headline locations only offer a landing page
            "select": "doi,best_oa_location,primary_location,open_access,locations",
        }
        openalex_auth(params)
        try:
            data = _http_get_json(OPENALEX_SEARCH_URL, params,
                                  timeout=timeout, retries=2)
        except Exception:
            continue        # an OA lookup is a bonus, not a requirement
        for work in data.get("results") or []:
            doi, url = _doi_key(work), _oa_pdf_url(work)
            if doi and url:
                out.setdefault(doi, url)
    return out


def search_google_scholar(query, limit=20):
    """Query Google Scholar through SerpAPI, the de-facto third-party API.

    Google has never offered an official Scholar API and scholar.google.com
    blocks direct scraping (HTTP 403 + CAPTCHA), so Scholar data is only
    reachable via a paid proxy such as SerpAPI. Requires SERPAPI_KEY in .env;
    without it this raises and callers should degrade to the other sources.
    """
    if not SERPAPI_KEY:
        raise RuntimeError(
            "SERPAPI_KEY not set: Google Scholar has no official API and "
            "scholar.google.com blocks direct scraping; add a SerpAPI key "
            "(https://serpapi.com) to .env to enable this source"
        )
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": min(limit, 20),
        "hl": "en",
    }
    data = _http_get_json(SERPAPI_SEARCH_URL, params)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"SerpAPI error: {data['error']}")

    papers = []
    for r in data.get("organic_results", []):
        pub = r.get("publication_info") or {}
        summary = pub.get("summary") or ""
        m = re.search(r"\b(?:1[89]\d{2}|20\d{2})\b", summary)
        venue = ""
        if " - " in summary:
            parts = [s.strip() for s in summary.split(" - ")]
            for i, part in enumerate(parts):
                if re.search(r"\b(?:1[89]\d{2}|20\d{2})\b", part):
                    # segment containing the year; strip year/punct, prefer it
                    base = re.sub(r"\b(?:1[89]\d{2}|20\d{2})\b", "", part).strip(" .,;")
                    if base and "," not in base:  # avoid author lists
                        venue = base
                    elif i > 0 and "," not in parts[i - 1]:
                        venue = parts[i - 1]
                    elif i + 1 < len(parts) and len(parts[i + 1]) > 2:
                        venue = parts[i + 1]
                    break
        authors = [a.get("name", "") for a in (pub.get("authors") or [])]
        if not authors and summary:  # fall back to parsing the "A, B, C - 1997 - venue" summary
            authors = [s.strip() for s in summary.split("-", 1)[0].split(",") if s.strip()]
        cited_by = ((r.get("inline_links") or {}).get("cited_by") or {}).get("total")
        pdf_url = ""
        for res in r.get("resources") or []:
            if (res.get("file_format") or "").upper() == "PDF" and res.get("link"):
                pdf_url = res["link"]
                break
        papers.append(
            _normalize_paper(
                title=r.get("title"),
                abstract=r.get("snippet"),
                year=int(m.group(0)) if m else None,
                citations=cited_by,
                url=r.get("link"),
                authors=authors,
                source="google_scholar",
                pdf_url=pdf_url,
                venue=venue,
            )
        )
    return papers


# ------------------------------------------------------------ registration
# Built-in sources. Add a new database by registering it here (or from another
# module) — no other code has to change.
# S2 allows roughly 1 request/second: stagger the starts but let slow responses
# overlap. OpenAlex is happy with more; SerpAPI stays one-at-a-time.
# Only the sources whose function takes `sort` are used for the citation-ranked
# pass (semantic_scholar via its bulk endpoint, openalex via cited_by_count).
register_source("semantic_scholar", search_semantic_scholar, "Semantic Scholar",
                max_concurrency=3 if S2_API_KEY else 1,
                min_gap=1.0 if S2_API_KEY else 3.0,
                timeout=20, retries=2)
register_source("openalex", search_openalex, "OpenAlex",
                max_concurrency=2, min_gap=0.4, timeout=25, retries=2)
register_source("google_scholar", search_google_scholar, "Google Scholar (SerpAPI)",
                available=bool(SERPAPI_KEY), max_concurrency=1, min_gap=0.5,
                timeout=25, retries=2)
# second tier: added for coverage, no key required. arXiv asks for >=3s between
# requests (one connection at a time); the other two tolerate overlapping reads.
register_source("arxiv", search_arxiv, "arXiv (preprints)",
                max_concurrency=1, min_gap=3.0, timeout=20, retries=1)
register_source("europe_pmc", search_europe_pmc, "PubMed / Europe PMC",
                max_concurrency=2, min_gap=0.5, timeout=20, retries=2)
register_source("crossref", search_crossref, "Crossref",
                max_concurrency=2, min_gap=0.3, timeout=20, retries=2)


def _classic_keep(papers, min_citations=CLASSIC_MIN_CITATIONS):
    """Cut a citation-ranked hit list down to the field's landmark papers.

    Anything at or above `min_citations` is a genuine anchor. When nothing
    clears the bar (young or niche topic, or a source whose counts are mostly
    zero) the single most cited hit is kept, so the query still contributes
    its local anchor instead of nothing.
    """
    kept = [p for p in papers if (p.get("citations") or 0) >= min_citations]
    if kept:
        return kept
    best = max(papers, key=lambda p: p.get("citations") or 0, default=None)
    return [best] if best else []


def search_classics(queries, limit=CLASSIC_PER_QUERY, filters=None,
                    min_citations=CLASSIC_MIN_CITATIONS):
    """Citation-ranked companion pass over the same queries.

    Relevance ranking is dominated by whatever is new and well indexed, so a
    pool built only from it tends to date the whole review: the papers a field
    is actually measured against never show up. This pass asks the sources
    that can sort by citation count (Semantic Scholar bulk, OpenAlex,
    Crossref) for their most cited matches and keeps the landmarks.

    Returns the same shape as search_many_detailed, with `papers` already cut
    to the landmarks and `classic` set on every item.
    """
    queries = [q for q in queries if q.strip()]
    if not queries:
        return []
    details = search_many_detailed(queries, limit=limit, filters=filters, sort="citations")
    for d in details:
        d["papers"] = _classic_keep(d["papers"], min_citations)
        d["classic"] = True
    return details


def search(query, limit=20, source="auto", filters=None):
    """Unified entry point.

    source="auto" (default) queries every available source concurrently and
    merges the hits (Google Scholar joins automatically when SERPAPI_KEY is
    configured). Any registered source name restricts to that single source.
    Optional `filters` {"min_year", "max_year", "review_only", "venue"}
    post-filters results consistently across every source.
    """
    if source != "auto":
        fn = _source_fn(source)
        if fn is None:
            raise ValueError(f"unknown source: {source}")
        return _apply_filters(fn(query, limit), filters)

    slot = _query_sources_many([query], limit, filters)[0]
    papers = _merge_hits(slot["results"])
    if not papers and slot["errors"]:
        raise RuntimeError("all search sources returned nothing: " + "; ".join(slot["errors"]))
    return papers


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "retrieval augmented generation survey"
    results = search(q, limit=5)
    sources = sorted({p["source"] for p in results})
    print(f"query: {q}\nresults: {len(results)} papers | sources: {', '.join(sources)}\n")
    for i, p in enumerate(results, 1):
        print(f"[{i}] {p['title']} ({p['year']}, cited {p['citations']})")
        print(f"    {', '.join(p['authors'][:3])}")
        print(f"    {p['url']}")
        if p["abstract"]:
            print(f"    {p['abstract'][:200]}...")
        print()
