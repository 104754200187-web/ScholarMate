"""Full-text retrieval module (pure stdlib).

Downloads paper PDFs (arXiv / open-access), extracts text without any third
party library, splits it into overlapping chunks, and ranks chunks with BM25
for a query. Used by the per-sentence evidence check to go beyond abstracts,
and usable standalone:

    python3 fulltext.py "retrieval augmented generation" --arxiv 2005.11401
    python3 fulltext.py "hallucination" --pdf https://arxiv.org/pdf/2311.05232
"""

import contextvars

import atomicio
import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

FULLTEXT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fulltext_cache")
ENABLED = os.environ.get("FULLTEXT", "1") != "0"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
TOP_PASSAGES = 2
MAX_PASSAGE_CHARS = 320
MIN_TEXT_CHARS = 200

_UA = "Mozilla/5.0 (X11; Linux x86_64) research-agent/0.1 (literature agent)"
_ARXIV_ABS = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#]+)")
_memo = {}  # cache key -> (text, note); avoids re-downloading within a run
# DOI -> openly available PDF URL, filled once per run by the caller from
# OpenAlex (search.openalex_oa_pdf_urls). Merged rather than replaced: a DOI
# names one paper no matter which run asked, so a second run's map can only add
# addresses. Used for papers whose own URL is missing or known to fail only.
_oa_urls = {}

# Latency controls. A slow PDF used to cost up to timeout*retries (~180s); with
# a shorter timeout and fewer retries the worst case is ~50s, and downloads run
# concurrently (see prefetch_fulltexts).
DOWNLOAD_TIMEOUT = int(os.environ.get("FULLTEXT_TIMEOUT", "25"))
DOWNLOAD_RETRIES = int(os.environ.get("FULLTEXT_RETRIES", "2"))
PREFETCH_WORKERS = int(os.environ.get("FULLTEXT_WORKERS", "4"))
DEADLINE_S = int(os.environ.get("FULLTEXT_DEADLINE_S", "300"))
ARXIV_MIN_GAP = 3.0  # arXiv asks for >=3s between requests
# Guard rails for pathological PDFs: a large, text-poor file (scanned/figure
# heavy) can otherwise pin the pure-stdlib extractor for minutes. Measured on
# figure-heavy publisher PDFs: 0.9MB -> 1s, 8MB -> 102s, so the wall clock --
# not the byte count -- is what has to be capped.
MAX_PDF_BYTES = int(os.environ.get("FULLTEXT_MAX_BYTES", "8000000"))
EXTRACT_TIMEOUT = int(os.environ.get("FULLTEXT_EXTRACT_TIMEOUT", "20"))
# A paywalled (403) or too-slow PDF stays unavailable for days, and re-paying
# 20s+ for it on every run of every question is pure waste: remember the
# failures on disk. 0 disables the markers.
SKIP_TTL = float(os.environ.get("FULLTEXT_SKIP_TTL", str(7 * 86400)))
# A network stumble is not a paywall: a host that timed out today may answer
# tomorrow, so those markers expire much sooner than a 403 / unreadable PDF.
SKIP_TTL_TRANSIENT = float(
    os.environ.get("FULLTEXT_SKIP_TTL_TRANSIENT", str(6 * 3600)))
EXTRACT_CONCURRENCY = int(os.environ.get("FULLTEXT_EXTRACT_CONCURRENCY", "2"))

STATS = {
    "memo_hits": 0,        # served from this run's memory
    "cache_hits": 0,       # served from fulltext_cache/ on disk
    "downloads": 0,
    "download_failed": 0,
    "oa_fallbacks": 0,     # attempts at the OpenAlex OA copy: the address the
                           # paper would otherwise never have had, or its second
                           # one after the first failed
    "html_resolved": 0,    # PDF reached through the landing page's own link
    "deadline_skips": 0,
    "too_large_skips": 0,  # PDF bigger than MAX_PDF_BYTES
    "extract_timeouts": 0, # extraction exceeded EXTRACT_TIMEOUT
    "extract_busy": 0,     # no free extraction slot, fell back to abstract
    "skip_markers": 0,     # known-bad PDF skipped without a download
    "bytes": 0,
    "download_seconds": 0.0,
    "extract_seconds": 0.0,
}
_extract_sem = threading.Semaphore(max(1, EXTRACT_CONCURRENCY))
_stats_lock = threading.Lock()
_arxiv_lock = threading.Lock()
_arxiv_last = [0.0]
_key_locks = {}
_key_locks_guard = threading.Lock()
# the full-text cap of the run executing in this thread's context (see
# set_deadline); a process-wide slot would let one run shorten another's stage
_deadline = contextvars.ContextVar("fulltext_deadline", default=None)


def reset_stats():
    with _stats_lock:
        for k in STATS:
            STATS[k] = 0.0 if isinstance(STATS[k], float) else 0
    _deadline.set(None)


def _bump(field, amount=1):
    with _stats_lock:
        STATS[field] = STATS.get(field, 0) + amount


def stats_summary():
    with _stats_lock:
        s = dict(STATS)
    total = (s["memo_hits"] + s["cache_hits"] + s["downloads"] + s["deadline_skips"]
             + s["too_large_skips"] + s["extract_timeouts"] + s["extract_busy"])
    s["total"] = total
    s["cache_hit_rate"] = round((s["memo_hits"] + s["cache_hits"]) / total, 2) if total else 0.0
    return s


def set_deadline(seconds):
    """Cap the whole full-text stage; later lookups fall back to abstracts.

    A context variable rather than a global: the cap belongs to the run that
    set it, and lookups run in pool threads, which inherit a copy of the
    setting run's context (see sentences.run).
    """
    _deadline.set((time.monotonic() + seconds) if seconds else None)


def clear_deadline():
    _deadline.set(None)


def _deadline_hit():
    at = _deadline.get()
    return at is not None and time.monotonic() > at


def _key_lock(key):
    with _key_locks_guard:
        lock = _key_locks.get(key)
        if lock is None:
            lock = _key_locks[key] = threading.Lock()
        return lock


# cache key -> {address: {"reason", "at", "ttl"}}, persisted next to the cache.
# Per address, because one paper can be reachable at two: the URL a search
# result carried and the open-access copy OpenAlex knows. Storing one marker
# per paper let the second failure erase the first, so the expensive address
# that already 403'd came back every other run.
_SKIP_ANY = "*"          # a marker that applies to whatever address is asked
_skips = {}
_skips_loaded = False
_skip_lock = threading.Lock()


def _skips_path():
    return os.path.join(FULLTEXT_CACHE, "_skips.json")


def _load_skips():
    global _skips, _skips_loaded
    if _skips_loaded:
        return
    try:
        with open(_skips_path(), encoding="utf-8") as f:
            loaded = json.load(f)
        _skips = loaded if isinstance(loaded, dict) else {}
    except Exception:
        _skips = {}
    for key, rec in list(_skips.items()):
        _skips[key] = _migrated(rec)
    _skips_loaded = True


def _migrated(rec):
    """One paper's marker as {address: {"reason", "at", "ttl"}}.

    Files written before addresses were recorded hold a single flat record
    ({reason, at, ttl, url}); it is read as a marker for that one address, or
    -- when it names none -- for whatever address is asked about.
    """
    if not isinstance(rec, dict):
        return {}
    if "reason" in rec or "at" in rec:
        return {((rec.get("url") or "").strip() or _SKIP_ANY): {
            "reason": rec.get("reason") or "full text unavailable",
            "at": rec.get("at") or 0,
            "ttl": rec.get("ttl"),
        }}
    return {addr: slot for addr, slot in rec.items() if isinstance(slot, dict)}


def _fresh(slot):
    """The reason a marker still stands for, or "" once it has expired."""
    ttl = float(slot.get("ttl") or SKIP_TTL)
    if time.time() - float(slot.get("at") or 0) > ttl:
        return ""
    return slot.get("reason") or "full text unavailable"


def _save_skips():
    try:
        atomicio.write_json(_skips_path(), _skips)
    except Exception:
        pass


def _skip_reason(key, url=""):
    """Why this paper's full text was given up on recently ("" if it was not).

    Markers are per address: a publisher answering 403 for the URL a search
    result carried says nothing about the open-access copy OpenAlex knows, so
    that second address stays worth a try. Passing no `url` asks about the
    paper rather than one address (see is_known_bad).
    """
    if not SKIP_TTL:
        return ""
    with _skip_lock:
        _load_skips()
        slots = _skips.get(key)
        if not slots:
            return ""
        # the address asked about, then any marker that names no address
        order = dict.fromkeys([url, _SKIP_ANY]) if url else dict.fromkeys(slots)
        for addr in order:
            slot = slots.get(addr)
            if not slot:
                continue
            reason = _fresh(slot)
            if reason:
                return reason
            slots.pop(addr, None)   # expired: forget this address only
        if not slots:
            _skips.pop(key, None)
        _save_skips()
        return ""


def _mark_skip(key, reason, ttl=None, url=""):
    """Remember a terminal failure so the next run skips this PDF outright."""
    if not SKIP_TTL:
        return
    if ttl is None:
        ttl = SKIP_TTL_TRANSIENT if _is_connection_failure(reason) else SKIP_TTL
    if not ttl:
        return
    with _skip_lock:
        _load_skips()
        slots = _skips.setdefault(key, {})
        slots[((url or "").strip()[:500] or _SKIP_ANY)] = {
            "reason": str(reason)[:200], "at": time.time(), "ttl": ttl}
        _save_skips()


def _clear_skip(key):
    """Forget every marker for this paper -- one of its addresses worked."""
    with _skip_lock:
        _load_skips()
        if _skips.pop(key, None) is None:
            return
        _save_skips()


def _arxiv_wait(url):
    """Serialise arXiv downloads so concurrent workers keep the 3s gap."""
    if "arxiv.org" not in url:
        return
    with _arxiv_lock:
        gap = ARXIV_MIN_GAP - (time.monotonic() - _arxiv_last[0])
        if gap > 0:
            time.sleep(gap)
        _arxiv_last[0] = time.monotonic()


def _is_connection_failure(reason):
    """True when retrying cannot help: the host never answered."""
    reason = str(reason).lower()
    return any(k in reason for k in
               ("timed out", "timeout", "handshake", "connection reset",
                "connection refused", "unreachable"))


def _http_get(url, timeout=None, retries=None, max_bytes=None):
    """max_bytes overrides the module cap: text extraction is CPU-bound and
    needs a small one, while a caller that only stores the bytes (pdfpack) can
    afford a bigger file."""
    cap = MAX_PDF_BYTES if max_bytes is None else int(max_bytes)
    timeout = DOWNLOAD_TIMEOUT if timeout is None else timeout
    retries = DOWNLOAD_RETRIES if retries is None else retries
    retries = max(1, retries)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                declared = resp.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > cap:
                    raise RuntimeError(
                        f"pdf too large: {int(declared)} bytes > {cap}")
                data = resp.read(cap + 1)
                if len(data) > cap:
                    raise RuntimeError(f"pdf too large: >{cap} bytes")
                return data
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8) + (time.monotonic() % 1))
                continue
            raise
        except urllib.error.URLError as e:
            last_err = e
            reason = str(getattr(e, "reason", e))
            # one connection attempt, not two: a PDF host that times out or
            # fails the TLS handshake does not recover a second later, and the
            # retry doubles this paper's cost (25s -> 50s) in a stage that runs
            # these downloads in parallel against a shared wall clock
            if _is_connection_failure(reason):
                raise RuntimeError(f"download failed: {reason}") from None
            if attempt + 1 < retries:
                time.sleep(1 + attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {last_err}")


# ---------------------------------------------------------------- PDF text

def _unescape_pdf_string(s):
    out = bytearray()
    i = 0
    mapping = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f",
               "(": "(", ")": ")", "\\": "\\"}
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt in mapping:
                out.extend(mapping[nxt].encode("latin-1"))
                i += 2
                continue
            if "0" <= nxt <= "7":  # octal escape \ddd (ASCII 0-7 only)
                j = i + 1
                digits = ""
                while j < len(s) and j < i + 4 and "0" <= s[j] <= "7":
                    digits += s[j]
                    j += 1
                out.append(int(digits, 8) & 0xFF)
                i = j
                continue
        out.extend(c.encode("latin-1"))
        i += 1
    return bytes(out).decode("latin-1", "replace")


def _decode_content_stream(raw):
    """Extract text shown by text operators (Tj/TJ/'/\"), handling both
    parenthesized literals, hex-encoded strings, and TJ arrays (whose
    elements are numbers, not operators). Items are collected with their
    offsets and emitted in stream order."""
    items = []
    for m in re.finditer(rb"((?:\((?:\\.|[^()\\])*\))|(?:<[0-9A-Fa-f\s]+>))\s*(?:Tj|'|\")", raw):
        text = _decode_string(m.group(1))
        if not _is_garbage(text):
            items.append((m.start(), text))
    for m in re.finditer(rb"\[(.*?)\]\s*TJ", raw, re.DOTALL):
        arr = m.group(1)
        for tok in re.finditer(
            rb"\((?:\\.|[^()\\])*\)|<[0-9A-Fa-f\s]+>|[-+]?\d+(?:\.\d+)?", arr
        ):
            t = tok.group(0)
            if t.startswith(b"(") or t.startswith(b"<"):
                text = _decode_string(t)
                if not _is_garbage(text):
                    items.append((m.start(1) + tok.start(), text))
            else:
                try:
                    val = float(t)
                except ValueError:
                    continue
                if val <= -100:  # negative TJ adjustment beyond kerning = word space
                    items.append((m.start(1) + tok.start(), " "))
    items.sort()
    return "".join(text for _, text in items)


def _decode_string(lit):
    """Decode one PDF string operand (parenthesized or hex) to text."""
    if lit.startswith(b"("):
        return _unescape_pdf_string(lit[1:-1].decode("latin-1"))
    h = re.sub(rb"\s", b"", lit[1:-1])
    try:
        return bytes.fromhex(h.decode("ascii")).decode("latin-1", "replace")
    except ValueError:
        return ""


def _is_garbage(text):
    """Heuristic: reject decoded strings that are mostly non-word bytes
    (embedded fonts, glyph tables, binary noise)."""
    if not text:
        return True
    wordy = sum(1 for ch in text if ch.isalnum())
    return wordy / len(text) < 0.6


def _printable_ratio(text):
    """Fraction of characters that are printable text rather than control
    bytes; used to reject font/image streams decoded as garbage."""
    if not text:
        return 0.0
    good = sum(
        1 for ch in text
        if ch in " \t\r\n" or 32 <= ord(ch) < 127 or ord(ch) >= 0x80
    )
    return good / len(text)


def _decompress(stream):
    try:
        return zlib.decompress(stream)
    except Exception:
        pass
    d = zlib.decompressobj()
    try:
        return d.decompress(stream) + d.flush()
    except Exception:
        return b""


def extract_pdf_text(data):
    """Extract plain text from a PDF's content streams (FlateDecode or raw)."""
    texts = []
    for m in re.finditer(rb"stream\r?\n(.*?)(?:\r?\n)?endstream", data, re.DOTALL):
        stream = m.group(1)
        raw = _decompress(stream)
        if not raw:
            raw = stream
        text = _decode_content_stream(raw)
        if len(text) >= 40 and _printable_ratio(text) >= 0.85:
            texts.append(text)
    full = " ".join(texts)
    # drop duplicate and garbage sentences (PDFs often render text twice)
    seen, keep = set(), []
    for s in re.split(r"(?<=[.!?;])\s+", full):
        key = re.sub(r"\s+", " ", s).strip().lower()
        if not key or key in seen:
            continue
        if _printable_ratio(s) < 0.85:
            continue
        seen.add(key)
        keep.append(s)
    return " ".join(keep)


# ---------------------------------------------------------------- chunking

def chunk_text(text, max_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks at sentence boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?;。！？；])\s+", text)
    chunks, current, cur_len = [], [], 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if cur_len + len(s) > max_chars and current:
            chunks.append(" ".join(current))
            tail, tlen = [], 0
            for t in reversed(current):
                if tlen + len(t) > overlap:
                    break
                tail.append(t)
                tlen += len(t) + 1
            current = list(reversed(tail))
            cur_len = tlen
        current.append(s)
        cur_len += len(s) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------- BM25

def tokens(text):
    """English word tokens + CJK character bigrams (for Chinese queries)."""
    out = re.findall(r"[a-z0-9]+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


def bm25_scores(query, docs, k1=1.5, b=0.75):
    q = tokens(query)
    if not q or not docs:
        return [0.0] * len(docs)
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in q}
    scores = []
    for d in docs:
        counts = {}
        for t in d:
            counts[t] = counts.get(t, 0) + 1
        dl = len(d)
        s = 0.0
        for t in q:
            f = counts.get(t, 0)
            if f:
                s += idf.get(t, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def retrieve_passages(query, text, top_k=TOP_PASSAGES):
    chunks = chunk_text(text)
    if not chunks:
        return []
    scores = bm25_scores(query, [tokens(c) for c in chunks])
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in ranked[:top_k]:
        if scores[i] <= 0:
            break
        out.append({"score": round(scores[i], 3), "text": chunks[i][:MAX_PASSAGE_CHARS]})
    return out


# ---------------------------------------------------------------- fetch + cache

def pdf_url_for(paper):
    pdf = (paper.get("pdf_url") or "").strip()
    if pdf:
        return pdf
    aid = (paper.get("arxiv_id") or "").strip()
    if aid:
        return f"https://arxiv.org/pdf/{aid}"
    url = (paper.get("url") or "").strip()
    m = _ARXIV_ABS.search(url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    return ""


def doi_key(value):
    """Normalise a DOI from a paper dict or a raw string."""
    doi = value.get("doi") if isinstance(value, dict) else value
    doi = (doi or "").strip().lower()
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)


def set_oa_urls(mapping):
    """Remember the open-access PDF URLs a caller looked up (see sentences.py)."""
    if mapping:
        _oa_urls.update(mapping)


def alternative_pdf_url(paper):
    """A second address for this paper's PDF, or "" -- the OpenAlex OA copy.

    The per-sentence stage asks OpenAlex about the papers that need this (no
    URL of their own, or a URL that failed before) and hands the answer over
    with set_oa_urls. Addresses we already have are not repeated.
    """
    key = doi_key(paper)
    url = _oa_urls.get(key) if key else ""
    return url if url and url != pdf_url_for(paper) else ""


def is_known_bad(paper):
    """True when this paper's full text was given up on recently.

    The per-sentence stage uses it to decide which papers are worth asking
    OpenAlex about: a 403 on the URL a search result carried does not rule out
    an open-access copy somewhere else.
    """
    return bool(_skip_reason(_cache_key(paper)))


# Publishers answer a bot with a landing page long before a PDF: OUP / SAGE /
# Wiley / MDPI all advertise the real file in `<meta name="citation_pdf_url">`.
_CITATION_PDF = re.compile(
    rb"""<meta[^>]+name=["']citation_pdf_url["'][^>]*?content=["']([^"']+)["']""",
    re.I)
_CITATION_PDF_REV = re.compile(
    rb"""<meta[^>]+content=["']([^"']+)["'][^>]*?name=["']citation_pdf_url["']""",
    re.I)
_HTML_PDF_LINK = re.compile(rb"""href=["']([^"'\s]+\.pdf[^"'\s]*)["']""", re.I)


def pdf_link_from_html(data, base=""):
    """The PDF URL a landing page advertises, or "".

    Returns an absolute URL: the meta tag is both the cheapest and the most
    reliable thing to look for, and it is preferred over a bare `.pdf` link.
    """
    head = data[:200_000]
    m = _CITATION_PDF.search(head) or _CITATION_PDF_REV.search(head)
    if not m:
        m = _HTML_PDF_LINK.search(head)
    if not m:
        return ""
    url = m.group(1).decode("utf-8", "ignore").strip()
    url = url.replace("&amp;", "&")
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/") and base:
        m2 = re.match(r"(https?://[^/]+)", base)
        if m2:
            return m2.group(1) + url
    return url if url.startswith("http") else ""


def looks_like_pdf(data):
    """True when these bytes are a PDF rather than an HTML landing page."""
    return bool(data[:5].startswith(b"%PDF") or b"%PDF" in data[:1024])


def _cache_key(paper):
    aid = (paper.get("arxiv_id") or "").strip()
    if aid:
        return aid
    pdf = (paper.get("pdf_url") or "").strip()
    if pdf:
        return hashlib.sha1(pdf.encode()).hexdigest()[:16]
    return hashlib.sha1((paper.get("url") or "").encode()).hexdigest()[:16]


def _extract_guarded(data):
    """Extract text with a wall-clock cap.

    The pure-stdlib extractor cannot be interrupted, so it runs on a daemon
    thread; on timeout we stop waiting and fall back to the abstract. A slot
    semaphore bounds how many extractions can be in flight (and thus how many
    abandoned threads can pile up). Returns (text, note)."""
    if not _extract_sem.acquire(blocking=False):
        _bump("extract_busy")
        return "", "extraction busy"
    box = {}

    def work():
        try:
            box["text"] = extract_pdf_text(data)
        except Exception as e:  # pragma: no cover - defensive
            box["err"] = e
        finally:
            _extract_sem.release()

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(EXTRACT_TIMEOUT)
    if th.is_alive():
        _bump("extract_timeouts")
        return "", f"text extraction timed out after {EXTRACT_TIMEOUT}s"
    if "err" in box:
        raise box["err"]
    return box.get("text", ""), ""


def _download(key, url):
    """Download one candidate URL. Returns (data, error) -- never raises."""
    if _deadline_hit():
        _bump("deadline_skips")
        return None, "full-text stage deadline reached"
    _arxiv_wait(url)
    t0 = time.monotonic()
    try:
        data = _http_get(url)
    except Exception as e:
        _bump("download_failed")
        if "too large" in str(e):
            _bump("too_large_skips")
        return None, f"download failed: {e}"
    _bump("downloads")
    _bump("bytes", len(data))
    _bump("download_seconds", time.monotonic() - t0)
    return data, ""


def _read_one(key, path, url):
    """Fetch and extract one candidate URL's text.

    Returns (text, note): text is "" on failure and note says why. Terminal
    failures are remembered on disk against *this* URL, so a second address for
    the same paper stays worth a try on the next run.
    """
    data, err = _download(key, url)
    if err:
        _mark_skip(key, err, url=url)
        return "", err
    if not looks_like_pdf(data):
        # publishers hand a bot a landing page: the PDF is the link it carries
        inner = pdf_link_from_html(data, url)
        if not inner:
            err = "landing page carried no PDF link"
            _mark_skip(key, err, url=url)
            return "", err
        _bump("html_resolved")
        data2, err2 = _download(key, inner)
        if err2 or not looks_like_pdf(data2):
            err = err2 or "landing page did not yield a PDF"
            _mark_skip(key, err, url=url)
            return "", err
        data = data2
    if _deadline_hit():
        _bump("deadline_skips")
        return "", "full-text stage deadline reached"
    t1 = time.monotonic()
    text, note = _extract_guarded(data)
    _bump("extract_seconds", time.monotonic() - t1)
    if not text:
        # a per-paper failure (unreadable / too slow); "extraction busy" is
        # only a slot shortage, so it is retried by the next run
        if note != "extraction busy":
            # how fast the pure-stdlib extractor is depends on this machine,
            # not on the paper, so a timeout expires quickly
            ttl = SKIP_TTL_TRANSIENT if "timed out" in (note or "") else None
            _mark_skip(key, note or "text extraction failed", ttl=ttl, url=url)
        return "", note or "text extraction failed"
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < MIN_TEXT_CHARS:
        err = f"text extraction failed ({len(text)} chars)"
        _mark_skip(key, err, url=url)
        return "", err
    atomicio.write_text(path, text)
    return text, f"fulltext ({len(text)} chars)"


def get_fulltext(paper):
    """Return (text, note) for a paper. Text is cached in memory for the run
    and on disk under fulltext_cache/, so repeat questions are download-free.
    Safe to call from several threads; it never raises.

    Two addresses are tried at most: the one the paper came with, and -- when
    the caller handed one over with set_oa_urls -- the open-access copy OpenAlex
    knows for its DOI. That covers the two ways a paper ends up abstract-only
    without anyone reading it: no PDF URL in the record at all, and a publisher
    answering 403 to a bot at the URL the search result carried.
    """
    if not ENABLED:
        return "", "fulltext disabled"
    key = _cache_key(paper)
    if key in _memo:
        _bump("memo_hits")
        return _memo[key]
    with _key_lock(key):
        if key in _memo:  # another worker finished it while we waited
            _bump("memo_hits")
            return _memo[key]
        os.makedirs(FULLTEXT_CACHE, exist_ok=True)
        path = os.path.join(FULLTEXT_CACHE, f"{key}.txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                text = f.read()
            _memo[key] = (text, "cached")
            _bump("cache_hits")
            return _memo[key]
        alt = alternative_pdf_url(paper)   # OpenAlex's copy, "" when unknown
        candidates = list(dict.fromkeys([u for u in (pdf_url_for(paper), alt) if u]))
        if not candidates:
            _memo[key] = ("", "no full-text URL")
            return _memo[key]
        skipped = [_skip_reason(key, url) for url in candidates]
        if all(skipped):
            _bump("skip_markers")
            _memo[key] = ("", f"skipped, unavailable in a recent run: {skipped[0]}")
            return _memo[key]
        last_err = ""
        for i, url in enumerate(candidates):
            if skipped[i]:
                _bump("skip_markers")
                last_err = f"skipped, unavailable in a recent run: {skipped[i]}"
                continue
            if alt and url == alt:
                # also when the OA copy is the paper's only address: that is
                # the 84-of-243 case this lookup exists for
                _bump("oa_fallbacks")
            text, note = _read_one(key, path, url)
            if text:
                _clear_skip(key)
                _memo[key] = (text, note)
                return _memo[key]
            if "deadline reached" in note:
                _memo[key] = ("", note)
                return _memo[key]
            last_err = note
        _memo[key] = ("", last_err or "no full text")
        return _memo[key]


def prefetch_fulltexts(papers):
    """Warm the full-text cache for several papers concurrently.

    arXiv downloads stay serialised (3s politeness gap) while other sources run
    in a small thread pool, so a stage that used to be N serial downloads
    becomes ~N/PREFETCH_WORKERS rounds. Returns how many were scheduled."""
    todo = {}
    for p in papers:
        if not p or not (pdf_url_for(p) or alternative_pdf_url(p)):
            continue
        key = _cache_key(p)
        if key in _memo:
            continue
        if os.path.exists(os.path.join(FULLTEXT_CACHE, f"{key}.txt")):
            continue
        todo.setdefault(key, p)
    if not todo:
        return 0
    workers = max(1, min(PREFETCH_WORKERS, len(todo)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        # per-task context copy, so each fetch sees this run's deadline
        list(ex.map(lambda pp: contextvars.copy_context().run(get_fulltext, pp),
                    list(todo.values())))
    return len(todo)


def retrieve_for_claim(claim, paper, candidates=None):
    """Full-text passages from one paper relevant to a claim. Tries the claim
    text first; falls back to the paper title (covers Chinese claims against
    English full texts). `candidates` sets how many BM25 passages to return
    (defaults to TOP_PASSAGES)."""
    text, note = get_fulltext(paper)
    if not text:
        return [], note
    n = candidates or TOP_PASSAGES
    passages = retrieve_passages(claim["text"], text, top_k=n)
    if not passages:
        title = paper.get("title") or ""
        if title:
            passages = retrieve_passages(title, text, top_k=n)
    return passages, note


# ---------------------------------------------------------------- CLI

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Full-text retrieval for a query")
    ap.add_argument("query", help="the claim/query to retrieve passages for")
    ap.add_argument("--arxiv", help="arXiv paper id, e.g. 2005.11401")
    ap.add_argument("--pdf", help="direct PDF URL")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args(argv)

    global FULLTEXT_CACHE
    paper = {}
    if args.arxiv:
        paper = {"arxiv_id": args.arxiv, "title": f"arXiv {args.arxiv}"}
    elif args.pdf:
        paper = {"pdf_url": args.pdf}
    else:
        ap.error("provide --arxiv ID or --pdf URL")

    text, note = get_fulltext(paper)
    print(f"[fulltext] {note}")
    if not text:
        return 1
    chunks = chunk_text(text)
    passages = retrieve_passages(args.query, text, top_k=args.top)
    print(f"[fulltext] {len(chunks)} chunks, top {len(passages)} passages for: {args.query}")
    for i, p in enumerate(passages, 1):
        print(f"\n--- passage {i} (score {p['score']}) ---")
        print(p["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
