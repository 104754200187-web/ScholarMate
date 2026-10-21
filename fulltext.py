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
import ipaddress
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import OrderedDict
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
_memo = OrderedDict()  # cache key -> (text, note); bounded process-wide cache
_memo_lock = threading.Lock()
MEMO_MAX_ENTRIES = int(os.environ.get("FULLTEXT_MEMO_MAX_ENTRIES", "128"))
MEMO_MAX_CHARS = int(os.environ.get("FULLTEXT_MEMO_MAX_CHARS", str(32 * 1024 * 1024)))
# DOI -> openly available PDF URL, filled once per run by the caller from
# OpenAlex (search.openalex_oa_pdf_urls). Merged rather than replaced: a DOI
# names one paper no matter which run asked, so a second run's map can only add
# addresses. Used for papers whose own URL is missing or known to fail only.
_oa_urls = OrderedDict()
_oa_urls_lock = threading.Lock()
OA_URL_MAX_ENTRIES = int(os.environ.get("FULLTEXT_OA_URL_MAX_ENTRIES", "20000"))

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
# A FlateDecode stream can expand far beyond the PDF that carries it. Cap the
# decompressed bytes so one small file cannot make the extractor allocate
# hundreds of megabytes.
MAX_INFLATE_BYTES = int(os.environ.get("FULLTEXT_MAX_INFLATE_BYTES",
                                       str(32 * 1024 * 1024)))
# A paywalled (403) or too-slow PDF stays unavailable for days, and re-paying
# 20s+ for it on every run of every question is pure waste: remember the
# failures on disk. 0 disables the markers.
SKIP_TTL = float(os.environ.get("FULLTEXT_SKIP_TTL", str(7 * 86400)))
# A network stumble is not a paywall: a host that timed out today may answer
# tomorrow, so those markers expire much sooner than a 403 / unreadable PDF.
SKIP_TTL_TRANSIENT = float(
    os.environ.get("FULLTEXT_SKIP_TTL_TRANSIENT", str(6 * 3600)))
EXTRACT_CONCURRENCY = int(os.environ.get("FULLTEXT_EXTRACT_CONCURRENCY", "2"))
SKIP_MAX_ENTRIES = int(os.environ.get("FULLTEXT_SKIP_MAX_ENTRIES", "5000"))

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


def _memo_weight(value):
    text, note = value if isinstance(value, tuple) else (value, "")
    return len(text or "") + len(note or "")


def _memo_get(key):
    with _memo_lock:
        return _memo.get(key)


def _memo_has(key):
    with _memo_lock:
        return key in _memo


def _memo_put(key, text, note):
    """Store one result and evict oldest entries to keep both caps."""
    value = (text, note)
    with _memo_lock:
        _memo.pop(key, None)
        _memo[key] = value
        while _memo and (len(_memo) > max(0, MEMO_MAX_ENTRIES)
                         or sum(_memo_weight(v) for v in _memo.values())
                         > max(0, MEMO_MAX_CHARS)):
            _memo.popitem(last=False)
    return value


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
    changed = False
    for key, rec in list(_skips.items()):
        slots = _migrated(rec)
        fresh = {addr: slot for addr, slot in slots.items() if _fresh(slot)}
        if fresh:
            _skips[key] = fresh
        else:
            _skips.pop(key, None)
        if fresh != slots:
            changed = True
    if _trim_skips() or changed:
        _save_skips()
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


def _trim_skips():
    """Drop expired papers and keep only the newest SKIP_MAX_ENTRIES."""
    changed = False
    for key, slots in list(_skips.items()):
        fresh = {addr: slot for addr, slot in slots.items() if _fresh(slot)}
        if fresh != slots:
            changed = True
        if fresh:
            _skips[key] = fresh
        else:
            _skips.pop(key, None)
    cap = max(0, SKIP_MAX_ENTRIES)
    if len(_skips) > cap:
        def newest(slots):
            return max((float(slot.get("at") or 0) for slot in slots.values()),
                       default=0.0)
        keep = set(sorted(_skips, key=lambda k: newest(_skips[k]), reverse=True)[:cap])
        for key in list(_skips):
            if key not in keep:
                _skips.pop(key, None)
        changed = True
    return changed


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
        _trim_skips()
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


_LOCAL_HOST_SUFFIXES = (
    ".localhost", ".local", ".internal", ".home.arpa",
    # public wildcard-DNS domains commonly used to point a name at loopback
    ".nip.io", ".sslip.io", ".xip.io", ".localtest.me", ".lvh.me", ".vcap.me",
)


def _validate_url(url):
    """Return a URL a PDF download is allowed to touch, or raise ValueError.

    Only http/https is accepted: urllib would otherwise happily read file://
    and other local schemes when a search result or a landing page supplies the
    address. Literal loopback/private/link-local addresses are rejected here,
    and the redirect handler applies the same check to every hop.

    Hostnames are deliberately not resolved locally: this deployment reaches
    the internet through a proxy, where resolving the target host would add a
    DNS dependency (and fail) for URLs the proxy can fetch. The local-name and
    wildcard-DNS suffixes above cover the usual non-literal bypasses.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    url = url.strip()
    if any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise ValueError("URL contains a control character")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {parts.scheme or '(none)'}")
    if parts.username or parts.password:
        raise ValueError("URL must not carry credentials")
    host = (parts.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError("URL has no host")
    if host == "localhost" or host.endswith(_LOCAL_HOST_SUFFIXES):
        raise ValueError(f"local host is not allowed: {host}")
    try:
        port = parts.port
    except ValueError as e:
        raise ValueError(f"invalid URL port: {e}") from None
    if port not in (None, 80, 443):
        raise ValueError(f"URL port is not allowed: {port}")
    # Numeric hosts that ipaddress cannot parse (127.1, 2130706433, 0x7f000001)
    # are alternate spellings a resolver may still turn into a private address.
    numeric = re.fullmatch(r"[0-9.]+", host) or host.startswith("0x")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if numeric or ":" in host or "%" in host or re.search(r"\s", host):
            raise ValueError(f"ambiguous or invalid host: {host}") from None
    else:
        if not ip.is_global:
            raise ValueError(f"non-public IP address is not allowed: {host}")
    return url


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _http_get(url, timeout=None, retries=None, max_bytes=None):
    """max_bytes overrides the module cap: text extraction is CPU-bound and
    needs a small one, while a caller that only stores the bytes (pdfpack) can
    afford a bigger file."""
    url = _validate_url(url)
    cap = MAX_PDF_BYTES if max_bytes is None else int(max_bytes)
    timeout = DOWNLOAD_TIMEOUT if timeout is None else timeout
    retries = DOWNLOAD_RETRIES if retries is None else retries
    retries = max(1, retries)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    last_err = None
    for attempt in range(retries):
        try:
            opener = urllib.request.build_opener(_SafeRedirectHandler())
            with opener.open(req, timeout=timeout) as resp:
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


def _inflate_bounded(stream, wbits):
    """Inflate one stream with a hard output cap.

    Returns (data, too_large). `decompress(..., max_length)` leaves the rest
    in `unconsumed_tail` when output hits the limit, so no allocation beyond
    the cap occurs before we can reject it.
    """
    cap = max(0, MAX_INFLATE_BYTES)
    d = zlib.decompressobj(wbits)
    try:
        out = d.decompress(stream, cap + 1)
        if len(out) > cap or d.unconsumed_tail:
            return b"", True
        tail = d.flush()
    except zlib.error:
        return b"", False
    out += tail
    if len(out) > cap:
        return b"", True
    return out, False


def _decompress(stream):
    # zlib wrapper first, then raw deflate (some PDFs omit the wrapper).
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        data, too_large = _inflate_bounded(stream, wbits)
        if too_large:
            return b""
        if data:
            return data
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
    if not mapping:
        return
    with _oa_urls_lock:
        for key, url in mapping.items():
            _oa_urls[key] = url
            _oa_urls.move_to_end(key)
        while len(_oa_urls) > max(0, OA_URL_MAX_ENTRIES):
            _oa_urls.popitem(last=False)


def alternative_pdf_url(paper):
    """A second address for this paper's PDF, or "" -- the OpenAlex OA copy.

    The per-sentence stage asks OpenAlex about the papers that need this (no
    URL of their own, or a URL that failed before) and hands the answer over
    with set_oa_urls. Addresses we already have are not repeated.
    """
    key = doi_key(paper)
    if not key:
        return ""
    with _oa_urls_lock:
        url = _oa_urls.get(key, "")
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
        # A normal arXiv id is a safe single path component and keeps the
        # cache readable. Anything else (a slash, "..", an absolute path) is
        # hashed so it can never escape FULLTEXT_CACHE.
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", aid):
            return aid
        return hashlib.sha1(aid.encode()).hexdigest()[:16]
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
    hit = _memo_get(key)
    if hit is not None:
        _bump("memo_hits")
        return hit
    with _key_lock(key):
        hit = _memo_get(key)
        if hit is not None:  # another worker finished it while we waited
            _bump("memo_hits")
            return hit
        os.makedirs(FULLTEXT_CACHE, exist_ok=True)
        path = os.path.join(FULLTEXT_CACHE, f"{key}.txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                text = f.read()
            value = _memo_put(key, text, "cached")
            _bump("cache_hits")
            return value
        alt = alternative_pdf_url(paper)   # OpenAlex's copy, "" when unknown
        candidates = list(dict.fromkeys([u for u in (pdf_url_for(paper), alt) if u]))
        if not candidates:
            return _memo_put(key, "", "no full-text URL")
        skipped = [_skip_reason(key, url) for url in candidates]
        if all(skipped):
            _bump("skip_markers")
            return _memo_put(key, "", f"skipped, unavailable in a recent run: {skipped[0]}")
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
                return _memo_put(key, text, note)
            if "deadline reached" in note:
                return _memo_put(key, "", note)
            last_err = note
        return _memo_put(key, "", last_err or "no full text")


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
        if _memo_has(key):
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
