"""Full-text fetch path: two addresses, landing-page resolution, per-URL skips."""
import json, os, shutil, sys, tempfile, urllib.error
import _paths  # noqa: F401
import fulltext, sentences, search

fails = []
def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + repr(extra)))
    if not cond: fails.append(name)

TMP = tempfile.mkdtemp(prefix="fttest_")
PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n%%EOF\n"
LONG = "alpha beta gamma delta " * 40          # > MIN_TEXT_CHARS

class Net:
    """Stands in for fulltext._http_get: routes URLs to bytes or errors."""
    def __init__(self): self.routes, self.calls = {}, []
    def get(self, url, **kw):
        self.calls.append(url)
        r = self.routes.get(url)
        if r is None: raise AssertionError("unexpected download: " + url)
        if isinstance(r, Exception): raise r
        return r

def fresh(net, extract=None):
    fulltext.FULLTEXT_CACHE = TMP
    os.makedirs(TMP, exist_ok=True)
    for f in os.listdir(TMP):
        p = os.path.join(TMP, f)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    fulltext._memo.clear(); fulltext._skips.clear(); fulltext._skips_loaded = True
    fulltext._oa_urls.clear(); fulltext.reset_stats(); fulltext.clear_deadline()
    fulltext._arxiv_wait = lambda url: None
    fulltext._http_get = net.get
    fulltext._extract_guarded = extract or (lambda data: (LONG, ""))

HTTP403 = urllib.error.HTTPError("http://x", 403, "Forbidden", None, None)

# ---- 1. a landing page's own link is followed to the PDF -------------------
HTML = (b'<html><head><meta name="citation_pdf_url" '
        b'content="https://pub.example/real.pdf"></head><body>...</body></html>')
net = Net()
net.routes["https://pub.example/article"] = HTML
net.routes["https://pub.example/real.pdf"] = PDF
fresh(net)
text, note = fulltext.get_fulltext({"title": "T", "pdf_url": "https://pub.example/article"})
check("a landing page is followed to the PDF it advertises",
      text.startswith("alpha") and note.startswith("fulltext"), (text[:20], note))
check("counted as html_resolved", fulltext.STATS["html_resolved"] == 1, fulltext.STATS)
check("both hops were requested, in order",
      net.calls == ["https://pub.example/article", "https://pub.example/real.pdf"], net.calls)

# ---- 2. the landing-page parser itself ------------------------------------
cases = [
    ("meta tag, name first",
     b'<meta name="citation_pdf_url" content="https://a/b.pdf">', "http://x/", "https://a/b.pdf"),
    ("meta tag, content first",
     b'<meta content="https://a/b.pdf" name="citation_pdf_url">', "http://x/", "https://a/b.pdf"),
    ("single quotes",
     b"<meta name='citation_pdf_url' content='https://a/b.pdf'>", "http://x/", "https://a/b.pdf"),
    ("protocol-relative",
     b'<meta name="citation_pdf_url" content="//cdn.a/b.pdf">', "http://x/", "https://cdn.a/b.pdf"),
    ("root-relative, made absolute against the page",
     b'<meta name="citation_pdf_url" content="/dl/b.pdf">', "https://a/x/9", "https://a/dl/b.pdf"),
    ("bare .pdf link when there is no meta tag",
     b'<a href="/files/paper.pdf">PDF</a>', "https://a/x/9", "https://a/files/paper.pdf"),
    ("&amp; unescaped",
     b'<meta name="citation_pdf_url" content="https://a/b.pdf?x=1&amp;y=2">',
     "http://x/", "https://a/b.pdf?x=1&y=2"),
    ("nothing to find", b"<html><body>nope</body></html>", "http://x/", ""),
]
for label, html, base, want in cases:
    got = fulltext.pdf_link_from_html(html, base)
    check("pdf_link_from_html: " + label, got == want, (got, want))

check("looks_like_pdf accepts a PDF", fulltext.looks_like_pdf(PDF))
check("looks_like_pdf rejects a landing page", not fulltext.looks_like_pdf(HTML))

# ---- 3. no address of its own: the OpenAlex copy is tried ------------------
net = Net()
net.routes["https://oa.example/paper.pdf"] = PDF
fresh(net)
paper = {"title": "T", "doi": "10.1/x", "url": "https://doi.org/10.1/x"}
fulltext.set_oa_urls({"10.1/x": "https://oa.example/paper.pdf"})
text, note = fulltext.get_fulltext(paper)
check("a paper with no PDF URL is still attempted via its OA copy",
      text.startswith("alpha"), (text[:20], note))
check("counted as an OA fallback", fulltext.STATS["oa_fallbacks"] == 1, fulltext.STATS)
check("exactly that one address was requested",
      net.calls == ["https://oa.example/paper.pdf"], net.calls)

# the same paper without the OA map: no address, so no attempt at all
net2 = Net()
fresh(net2)
text, note = fulltext.get_fulltext(paper)
check("without a second address there is nothing to try",
      note == "no full-text URL" and net2.calls == [], (note, net2.calls))
check("'no full-text URL' is not a fetch failure", not sentences.fetch_failed(note), note)

# ---- 4. skips are recorded per address, not per paper ----------------------
net = Net()
net.routes["https://pub.example/own.pdf"] = HTTP403
net.routes["https://repo.example/copy.pdf"] = PDF
fresh(net)
paper = {"title": "T", "doi": "10.1/y", "pdf_url": "https://pub.example/own.pdf"}
fulltext.set_oa_urls({"10.1/y": "https://repo.example/copy.pdf"})
text, note = fulltext.get_fulltext(paper)
check("403 on the publisher URL falls through to the OA copy",
      text.startswith("alpha") and fulltext.STATS["oa_fallbacks"] == 1, (text[:20], note))
check("both addresses were tried", len(net.calls) == 2, net.calls)

key = fulltext._cache_key(paper)
fulltext._mark_skip(key, "download failed: HTTP Error 403: Forbidden",
                    url="https://pub.example/own.pdf")
check("the marker names the address that failed",
      "403" in (fulltext._skip_reason(key, "https://pub.example/own.pdf") or ""))
check("but not the paper's other address",
      fulltext._skip_reason(key, "https://repo.example/copy.pdf") == "")
check("is_known_bad still flags the paper (any address failing counts)",
      fulltext.is_known_bad(paper))

legacy = fulltext._cache_key({"title": "L", "pdf_url": "https://old.example/a.pdf"})
fulltext._mark_skip(legacy, "download failed: HTTP Error 403: Forbidden")
check("a marker from before URLs were recorded blocks every address",
      bool(fulltext._skip_reason(legacy, "https://other.example/z.pdf")))

# ---- 5. every address marked -> skipped, and that is a fetch failure -------
net = Net()
fresh(net)
paper = {"title": "T", "doi": "10.1/z", "pdf_url": "https://pub.example/own.pdf"}
fulltext.set_oa_urls({"10.1/z": "https://repo.example/copy.pdf"})
k = fulltext._cache_key(paper)
fulltext._mark_skip(k, "download failed: HTTP Error 403: Forbidden",
                    url="https://pub.example/own.pdf")
fulltext._mark_skip(k, "download failed: HTTP Error 403: Forbidden",
                    url="https://repo.example/copy.pdf")
text, note = fulltext.get_fulltext(paper)
check("both addresses marked: no download is attempted",
      text == "" and net.calls == [], (text, net.calls))
check("the note says it was skipped as recently unavailable",
      "unavailable in a recent run" in note, note)
check("and the UI treats it as 无法核对, not 不支持",
      sentences.fetch_failed(note), note)

# ---- 6. every note this path can emit is classified ------------------------
notes = [
    "download failed: HTTP Error 403: Forbidden",
    "landing page carried no PDF link",
    "landing page did not yield a PDF",
    "text extraction timed out after 20s",
    "text extraction failed (12 chars)",
    "extraction busy",
    "full-text stage deadline reached",
    "skipped, unavailable in a recent run: download failed: x",
]
missing = [n for n in notes if not sentences.fetch_failed(n)]
check("every 原文未取到 note is recognised by the UI classifier", not missing, missing)
check("an abstract-only verdict is still 不支持, not a fetch failure",
      not sentences.fetch_failed("no full-text URL") and
      not sentences.fetch_failed("abstract only"))

# a landing page with no link: the paper is unreadable, not unsupported
net = Net()
net.routes["https://pub.example/article"] = b"<html>no pdf here</html>"
fresh(net)
text, note = fulltext.get_fulltext({"title": "T", "pdf_url": "https://pub.example/article"})
check("a landing page with no link yields no text",
      text == "" and note == "landing page carried no PDF link", (text, note))

# ---- 7. only the right papers are asked about -----------------------------
asked = []
def fake_oa(papers, batch=None, timeout=None):
    asked.append([p["title"] for p in papers])
    return {"10.1/a": "https://oa.example/a.pdf"}

real_oa = search.openalex_oa_pdf_urls
search.openalex_oa_pdf_urls = fake_oa
try:
    net = Net()
    fresh(net)
    good = {"title": "has url", "pdf_url": "https://pub.example/live.pdf"}
    noUrl = {"title": "no url", "doi": "10.1/a"}
    bad = {"title": "failed before", "doi": "10.1/b", "pdf_url": "https://pub.example/dead.pdf"}
    fulltext._mark_skip(fulltext._cache_key(bad), "download failed: HTTP Error 403: Forbidden")
    sentences._offer_alternative_urls([good, noUrl, bad])
    check("papers with no URL and papers that failed before are asked about; "
          "healthy ones are not",
          asked == [["no url", "failed before"]], asked)
    check("the answers reach fulltext",
          fulltext.alternative_pdf_url(noUrl) == "https://oa.example/a.pdf",
          fulltext._oa_urls)
    check("a healthy paper is not given a redundant second address",
          fulltext.alternative_pdf_url(good) == "")

    # prefetch must pick up a paper whose only address is the OA copy
    net.routes["https://oa.example/a.pdf"] = PDF
    scheduled = fulltext.prefetch_fulltexts([noUrl])
    check("prefetch schedules a paper that only has an OA copy", scheduled == 1, scheduled)
    check("and the text lands", bool(fulltext.get_fulltext(noUrl)[0]))
finally:
    search.openalex_oa_pdf_urls = real_oa

# ---- 8. alternative_pdf_url never repeats the paper's own address ----------
net = Net()
fresh(net)
same = {"title": "T", "doi": "10.1/same", "pdf_url": "https://a/same.pdf"}
fulltext.set_oa_urls({"10.1/same": "https://a/same.pdf"})
check("a duplicate second address is dropped", fulltext.alternative_pdf_url(same) == "",
      fulltext.alternative_pdf_url(same))
check("a paper with no DOI never gets one", fulltext.alternative_pdf_url({"title": "x"}) == "")

# ---- 9. markers written by the previous format still work ---------------
now = __import__("time").time()

def slot(addr, ttl=604800.0, reason="download failed: HTTP Error 403: Forbidden"):
    return {"reason": reason, "at": now, "ttl": ttl, "url": addr}

m = fulltext._migrated(slot("https://a/1.pdf"))
check("a flat marker that names an address becomes a marker for that address",
      list(m) == ["https://a/1.pdf"], m)
check("a flat marker that names no address keeps applying to any of them",
      list(fulltext._migrated({"reason": "403", "at": now, "ttl": 1.0})) == ["*"])
check("the new nested shape is read as it stands",
      fulltext._migrated({"https://a/1.pdf": {"reason": "x", "at": now, "ttl": 1.0}})
      == {"https://a/1.pdf": {"reason": "x", "at": now, "ttl": 1.0}})
check("junk does not crash the load", fulltext._migrated(None) == {}
      and fulltext._migrated({"https://a/1.pdf": "nonsense"}) == {})

legacy_path = os.path.join(TMP, "_legacy_skips.json")
with open(legacy_path, "w", encoding="utf-8") as f:
    json.dump({
        "k1": slot("https://a/1.pdf"),
        "k2": {"reason": "403", "at": now, "ttl": 604800.0},
        "k3": {"https://a/3.pdf": {"reason": "timed out", "at": now, "ttl": 1.0}},
    }, f)
real = json.load(open(legacy_path, encoding="utf-8"))
converted = {k: fulltext._migrated(v) for k, v in real.items()}
slots = [s for v in converted.values() for s in v.values()]
check(f"every one of the {len(real)} legacy markers survives the migration",
      len(slots) == len(real) and all(s.get("reason") for s in slots),
      (len(real), len(slots)))

# an expired address is forgotten without taking the paper's other one with it
fresh(Net())
k = "k" + str(now)
fulltext._mark_skip(k, "403", ttl=-1, url="https://a/dead.pdf")
fulltext._mark_skip(k, "403", url="https://a/live.pdf")
check("an expired marker does not hide a live one for another address",
      fulltext._skip_reason(k, "https://a/live.pdf") != "", fulltext._skips.get(k))
check("asking about the expired address reports it as worth trying",
      fulltext._skip_reason(k, "https://a/dead.pdf") == "")
check("...and drops just that address", "https://a/dead.pdf" not in fulltext._skips.get(k, {}),
      fulltext._skips.get(k))
check("while the sibling marker stays", fulltext._skip_reason(k, "https://a/live.pdf") != "")

shutil.rmtree(TMP, ignore_errors=True)
print()
print(("ALL OK" if not fails else "FAILURES: " + ", ".join(fails)))
sys.exit(1 if fails else 0)
