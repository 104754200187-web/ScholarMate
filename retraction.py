"""Retraction screening for the paper pool.

A retracted paper is the most damaging thing a literature agent can cite, and
none of our search sources reports it consistently: Semantic Scholar and
SerpAPI have no retraction field at all. So we resolve it separately, in two
cheap steps, and cache the verdict on disk:

1. OpenAlex ``is_retracted`` (built on Retraction Watch data), fetched in bulk
   -- a single request covers up to 50 DOIs.
2. Crossref ``update-to`` relations of type ``retraction``, one request per
   DOI, for papers OpenAlex does not know about.

Both run under a wall-clock budget (RETRACTION_DEADLINE_S, default 8s) and
degrade to "unknown" rather than delaying the answer; unresolved DOI lookups
are simply retried on the next run and papers without a DOI are reported as
unchecked. Pure stdlib.
"""

import contextvars
import atomicio
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed

import search
import verify

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
CROSSREF_WORK_URL = "https://api.crossref.org/works/"
CROSSREF_SEARCH_URL = "https://api.crossref.org/works"
TITLE_ROWS = 3
BATCH_DOIS = 50
DEADLINE_S = float(os.environ.get("RETRACTION_DEADLINE_S", "10"))
WORKERS = int(os.environ.get("RETRACTION_WORKERS", "6"))
CROSSREF_MAX = int(os.environ.get("RETRACTION_CROSSREF_MAX", "40"))
CACHE_TTL_S = float(os.environ.get("RETRACTION_CACHE_TTL_DAYS", "30")) * 86400
# "we looked but could not tell" ages out much faster than a firm verdict, so a
# paper retracted later still gets re-checked within a few days
UNRESOLVED_TTL_S = float(os.environ.get("RETRACTION_UNRESOLVED_TTL_DAYS", "3")) * 86400
CACHE_PATH = os.environ.get(
    "RETRACTION_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "retraction_cache.json"),
)
HTTP_TIMEOUT = 6
HTTP_RETRIES = 2
# a title probe is best-effort: never let one slow query hold the whole stage
PROBE_TIMEOUT = 6
# Crossref throttles anonymous bursts from a shared egress IP, so a probe gets
# one backoff retry: cheaper than losing the verdict and re-asking next run
PROBE_RETRIES = 2
PROBE_WORKERS = int(os.environ.get("RETRACTION_PROBE_WORKERS", "6"))
# Polite-pool address for Crossref; falls back to the OpenAlex one.
CROSSREF_MAILTO = os.environ.get("CROSSREF_MAILTO") or os.environ.get("OPENALEX_MAILTO")
# Titles are the one signal that works with no network at all: journals prefix
# or suffix the retraction marker in the published title itself.
_RETRACTED_TITLE_RE = re.compile(
    r"(?:^\W{0,3}retract(?:ed|ion)\b)"
    r"|(?:\bthis article has been retracted\b)"
    r"|(?:\(\s*retracted\s*\)\s*$)",
    re.IGNORECASE,
)


def title_signal(title):
    """True when the published title itself says the paper was retracted."""
    return bool(_RETRACTED_TITLE_RE.search(title or ""))

_CACHE_LOCK = threading.Lock()


def _clean_doi(doi):
    """Normalize a DOI to its bare form (10.xxxx/yyyy), or "" if unusable."""
    doi = (doi or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi if doi.startswith("10.") else ""


def paper_key(paper):
    """Cache key of one paper.

    arXiv-only papers have no publisher DOI, but they do have a registered
    DataCite DOI (10.48550/arXiv.<id>) which OpenAlex knows, so we use it.
    Google Scholar gives no identifier at all, so those fall back to a
    title-keyed entry screened through a Crossref bibliographic query.
    """
    doi = _clean_doi(paper.get("doi") or paper.get("url") or "")
    if not doi and (paper.get("arxiv_id") or "").strip():
        doi = _clean_doi("10.48550/arXiv." + paper["arxiv_id"].strip())
    if doi:
        return "doi:" + doi
    title = verify.normalize(paper.get("title") or "")
    return "title:" + title if title else ""


def _load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    with _CACHE_LOCK:
        try:
            atomicio.write_json(CACHE_PATH, cache, indent=0)
        except OSError:
            pass


def _fresh(rec):
    try:
        age = time.time() - float(rec.get("checked_at") or 0)
    except (TypeError, ValueError):
        return False
    ttl = CACHE_TTL_S if rec.get("definitive", True) else UNRESOLVED_TTL_S
    return age < ttl


def _apply(paper, rec):
    # a record that is not definitive means "we looked and could not tell"
    paper["retracted"] = bool(rec.get("retracted")) if rec.get("definitive", True) else None
    paper["retraction_source"] = rec.get("source") or ""
    paper["retraction_note"] = rec.get("note") or ""
    paper["retraction_doi"] = rec.get("retraction_doi") or ""
    if rec.get("retraction_date"):
        paper["retraction_date"] = rec["retraction_date"]


def _openalex_batch(dois):
    """One request resolves up to BATCH_DOIS DOIs; returns {doi: record}."""
    params = {
        "filter": "doi:" + "|".join("https://doi.org/" + d for d in dois),
        "select": "doi,is_retracted",
        "per-page": len(dois),
    }
    search.openalex_auth(params)   # same credential rule as the search path
    # one attempt only: a rate-limited OpenAlex must not eat the whole budget
    # and starve the Crossref fallback below.
    data = search.http_get_json(OPENALEX_WORKS_URL, params,
                                timeout=HTTP_TIMEOUT, retries=1)
    out = {}
    for w in data.get("results", []):
        doi = _clean_doi(w.get("doi"))
        if not doi:
            continue
        retracted = bool(w.get("is_retracted"))
        out[doi] = {
            "retracted": retracted,
            "source": "openalex",
            "definitive": True,
            "checked_at": time.time(),
            "note": "OpenAlex / Retraction Watch 标记为已撤稿" if retracted else "",
        }
    return out


def _notice_date(notice):
    updated = notice.get("updated") or {}
    parts = (updated.get("date-parts") or [[]])[0]
    return "-".join(str(x) for x in parts if x)


# Journals mark retractions in the title itself ("RETRACTED: ...",
# "Retraction—..."), which breaks naive title matching against the original
# record, so strip the marker before comparing.
_MARKER_RE = re.compile(r"^\W{0,3}(?:retract(?:ed|ion))\b\W*", re.IGNORECASE)


def _strip_marker(title):
    return _MARKER_RE.sub("", title or "").strip()


def _crossref_record(msg):
    """Map one Crossref work into our record shape.

    `updated-by` lists the notices that update this work (that is where Crossref
    carries the Retraction Watch relations); `update-to` points the other way,
    from a notice to the paper it retracts -- a work carrying it is the notice
    itself, which callers interpret differently.
    """
    notices = [
        u for u in (msg.get("updated-by") or [])
        if str(u.get("type") or "").lower() == "retraction"
    ]
    if notices:
        notice = notices[-1]
        date = _notice_date(notice)
        label = str(notice.get("label") or "Retraction").strip()
        return {
            "retracted": True,
            "source": "crossref",
            "definitive": True,
            "checked_at": time.time(),
            "retraction_doi": notice.get("DOI") or "",
            "retraction_date": date,
            "note": f"Crossref/Retraction Watch 撤稿记录（{label}{'，' + date if date else ''}）",
        }
    if any(str(u.get("type") or "").lower() == "retraction"
           for u in (msg.get("update-to") or [])):
        return {"retracted": False, "source": "crossref", "definitive": True,
                "is_notice": True, "checked_at": time.time(),
                "note": "该 DOI 本身是撤稿声明（被撤稿的是另一篇论文）"}
    return {"retracted": False, "source": "crossref", "definitive": True,
            "checked_at": time.time(), "note": ""}


def _crossref_one(doi, timeout=None):
    """Per-DOI Crossref lookup; raises on network failure (not cached then)."""
    params = {}
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO
    try:
        data = search.http_get_json(CROSSREF_WORK_URL + urllib.parse.quote(doi), params,
                                    timeout=timeout or HTTP_TIMEOUT, retries=HTTP_RETRIES)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # nothing to compare against: unresolved, never reported as clean
            return {"retracted": False, "source": "crossref",
                    "definitive": False, "checked_at": time.time(),
                    "note": "Crossref 无此 DOI 记录，未能与撤稿数据比对"}
        raise
    return _crossref_record((data or {}).get("message") or {})


def _probe_title(title, timeout=None):
    """Screen a paper by title, for the sources that give no identifier.

    Google Scholar (SerpAPI) returns no DOI at all, so a bibliographic query is
    the only way to screen those hits. Deliberately asymmetric: a fuzzy title
    match may only ever produce a *retraction*, never a clean bill of health,
    so a wrong match can never bless a paper.
    """
    if not (title or "").strip():
        return None
    params = {
        "query.bibliographic": title,
        "rows": TITLE_ROWS,
        "select": "DOI,title,updated-by,update-to",
    }
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO
    data = search.http_get_json(CROSSREF_SEARCH_URL, params,
                                timeout=timeout or PROBE_TIMEOUT, retries=PROBE_RETRIES)
    items = ((data or {}).get("message") or {}).get("items") or []
    wanted = _strip_marker(title)
    for item in items:
        candidate = (item.get("title") or [""])[0]
        if not verify.title_matches(wanted, _strip_marker(candidate)):
            continue
        record = _crossref_record(item)
        if record.get("retracted"):
            return record
        if record.get("is_notice"):
            # the matched work *is* the retraction notice for our paper
            return {
                "retracted": True,
                "source": "crossref-title",
                "definitive": True,
                "checked_at": time.time(),
                "retraction_doi": item.get("DOI") or "",
                "note": "Crossref 标题检索命中针对该论文的撤稿声明",
            }
    return None


def _batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _run_crossref(items, cache, deadline, budget, missing=None):
    """Concurrent per-DOI Crossref lookups; returns the budget consumed."""
    items = items[:max(0, budget)]
    if not items or time.monotonic() >= deadline:
        return 0
    workers = max(1, min(WORKERS, len(items)))
    # shrink the per-request timeout as the deadline approaches, so a request
    # still in flight cannot overrun the stage by a full timeout
    timeout = max(1.0, min(HTTP_TIMEOUT, deadline - time.monotonic()))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(contextvars.copy_context().run, _crossref_one, doi, timeout):
                   (key, paper) for key, doi, paper in items}
        try:
            for fut in as_completed(futures, timeout=max(0.5, deadline - time.monotonic())):
                key, paper = futures[fut]
                try:
                    record = fut.result()
                except Exception:
                    continue
                cache[key] = record
                if missing is not None and not record.get("definitive"):
                    missing.append((key, paper))
        except FuturesTimeout:
            pass
    return len(items)


def _run_title_probes(items, cache, deadline, budget):
    items = items[:max(0, budget)]
    if not items or time.monotonic() >= deadline:
        return
    workers = max(1, min(PROBE_WORKERS, len(items)))
    timeout = max(1.0, min(PROBE_TIMEOUT, deadline - time.monotonic()))
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(contextvars.copy_context().run, _probe_title, title, timeout):
                   key for key, title in items}
        try:
            for fut in as_completed(futures, timeout=max(0.5, deadline - time.monotonic())):
                key = futures[fut]
                try:
                    record = fut.result()
                except Exception as e:
                    failures.append(f"{type(e).__name__}: {e}")
                    continue
                if not record:
                    # remember "looked, could not identify" so we do not re-ask
                    # Crossref on every run; ages out in UNRESOLVED_TTL_S
                    record = {"retracted": False, "source": "crossref-title",
                              "definitive": False, "checked_at": time.time(),
                              "note": "无 DOI，Crossref 标题检索未确证撤稿状态"}
                cache[key] = record
        except FuturesTimeout:
            pass
    if failures:
        print(f"      [warn] {len(failures)}/{len(items)} title probes failed "
              f"(e.g. {failures[0]}); they will be retried next run")


def _resolve(pending, cache, deadline, on_progress):
    """Fill `cache` for the pending (key, paper) pairs within the deadline."""
    doi_items = [(k, p) for k, p in pending if k.startswith("doi:")]
    title_items = [(k, p) for k, p in pending if not k.startswith("doi:")]
    dois = [k[4:] for k, _ in doi_items]
    resolved = set()
    # the search layer may already have parked OpenAlex on an IP-level 429; no
    # point spending the screening budget rediscovering that
    chunks = [] if search.breaker_open("openalex") else list(_batched(dois, BATCH_DOIS))
    for chunk in chunks:
        if time.monotonic() >= deadline:
            break
        try:
            found = _openalex_batch(chunk)
        except Exception as e:
            print(f"      [warn] OpenAlex retraction lookup failed: {e}")
            if not CROSSREF_MAILTO and "429" in str(e):
                print("[hint] OpenAlex is rate limiting us; set OPENALEX_MAILTO in "
                      ".env to join the polite pool (retraction screening relies on it)")
            found = {}
        for doi, rec in found.items():
            cache["doi:" + doi] = rec
            resolved.add(doi)
        if on_progress:
            on_progress(f"OpenAlex 核对 {len(resolved)}/{len(dois)} 篇")

    budget = CROSSREF_MAX
    rest = [(k, k[4:], p) for k, p in doi_items if k[4:] not in resolved]
    unresolved = []
    budget -= _run_crossref(rest, cache, deadline, budget, missing=unresolved)

    # title probes cover the sources that never hand out a DOI (Google Scholar)
    # plus any DOI Crossref could not identify
    probes = [(k, p.get("title") or "") for k, p in title_items]
    probes += [(k, p.get("title") or "") for k, p in unresolved]
    _run_title_probes(probes, cache, deadline, budget)

    for key, paper in pending:
        rec = cache.get(key)
        if rec:
            _apply(paper, rec)


def summary(papers):
    """Counts for the UI / audit log."""
    retracted = [p for p in papers if p.get("retracted")]
    checked = [p for p in papers if p.get("retracted") is not None]
    return {
        "total": len(papers),
        "checked": len(checked),
        "retracted": len(retracted),
        "unchecked": len(papers) - len(checked),
        "clean": len(checked) - len(retracted),
        "retracted_titles": [p.get("title", "") for p in retracted],
        "retracted_notices": [
            p.get("retraction_doi") or p.get("doi") or "" for p in retracted
        ],
    }


def screen(papers, deadline_s=None, on_progress=None):
    """Annotate every paper in place with its retraction status.

    Sets ``retracted`` (True/False/None = unknown), ``retraction_source``,
    ``retraction_note`` and ``retraction_doi``. Returns a summary dict.
    """
    papers = papers or []
    deadline = time.monotonic() + (deadline_s if deadline_s is not None else DEADLINE_S)
    cache = _load_cache()
    # titles are free and unambiguous: "RETRACTED: ..." needs no network
    for p in papers:
        if title_signal(p.get("title")) and p.get("retraction_source") != "title":
            p["retracted"] = True
            p["retraction_source"] = "title"
            p["retraction_note"] = "标题即标记为撤稿（RETRACTED）"
    pending = []
    for p in papers:
        if p.get("retraction_source") == "title":
            continue  # already settled without a request
        key = paper_key(p)
        if not key:
            # no DOI -> nothing to match on; report as unknown, not as clean
            p.setdefault("retracted", None)
            p.setdefault("retraction_note", "缺少标题/DOI，无法核对撤稿状态")
            continue
        rec = cache.get(key)
        if rec and _fresh(rec):
            _apply(p, rec)
        else:
            pending.append((key, p))
    if pending:
        _resolve(pending, cache, deadline, on_progress)
        _save_cache(cache)
    return summary(papers)
