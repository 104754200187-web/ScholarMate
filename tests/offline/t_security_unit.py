"""Run-id, download-URL, cache-key and request-body boundaries -- offline."""
import io
import json
import os
import tempfile

import _paths  # noqa: F401
import fulltext
import web

fails = []


def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + repr(extra)))
    if not cond:
        fails.append(name)


VALID_ID = "20261021_120000_123456"

# ---- run ids and snapshot paths --------------------------------------------
check("a traversal id gets no path", web._run_path("../../README") is None)
check("an absolute id gets no path", web._run_path("/tmp/evil") is None)
check("a dotted id gets no path", web._run_path("20261021_120000_123456.json") is None)
check("a generated id maps inside history",
      web._run_path(VALID_ID) == os.path.join(web.HISTORY_DIR, VALID_ID + ".json"),
      web._run_path(VALID_ID))
check("an invalid id is not a job", web._get_job("paper") is None)

with tempfile.TemporaryDirectory() as td:
    old_history = web.HISTORY_DIR
    web.HISTORY_DIR = td
    try:
        check("save_history refuses a traversal id", web._run_path("../../escape") is None)
        web._save_history({"id": "../../escape", "papers": []})
        check("nothing was written outside the history directory",
              not os.path.exists(os.path.join(os.path.dirname(td), "escape.json")))
        snap_id = VALID_ID
        with open(os.path.join(td, snap_id + ".json"), "w", encoding="utf-8") as f:
            json.dump({"id": snap_id, "papers": [{"title": "P"}], "mode": "ask"}, f)
        check("a snapshot without an owner is refused to a browser",
              web._load_job_or_snapshot(snap_id, "a" * 32) is None)
        check("an internal caller can still read a legacy snapshot",
              web._load_job_or_snapshot(snap_id) is not None)
        with open(os.path.join(td, snap_id + ".json"), "w", encoding="utf-8") as f:
            json.dump({"id": snap_id, "owner": "a" * 32,
                       "papers": [{"title": "P"}], "mode": "ask"}, f)
        check("the matching owner can read the snapshot",
              web._load_job_or_snapshot(snap_id, "a" * 32) is not None)
    finally:
        web.HISTORY_DIR = old_history

# ---- download URL policy ---------------------------------------------------
blocked = [
    "file:///etc/passwd",
    "ftp://example.com/a.pdf",
    "http://localhost/a.pdf",
    "http://127.0.0.1/a.pdf",
    "http://127.1/a.pdf",
    "http://2130706433/a.pdf",
    "http://0x7f000001/a.pdf",
    "http://10.0.0.1/a.pdf",
    "http://169.254.169.254/latest/meta-data",
    "http://[::1]/a.pdf",
    "http://[fe80::1%25eth0]/a.pdf",
    "http://exa mple.com/a.pdf",
    "http://example.com/a.pdf\r\nX-Injected: 1",
    "http://example.local/a.pdf",
    "http://metadata.google.internal/a.pdf",
    "http://127.0.0.1.nip.io/a.pdf",
    "http://example.com:6379/a.pdf",
    "http://user:pass@example.com/a.pdf",
]
for url in blocked:
    try:
        fulltext._validate_url(url)
        check("blocked URL rejected: " + url, False, "accepted")
    except ValueError:
        check("blocked URL rejected: " + url, True)

check("a normal https PDF URL is allowed",
      fulltext._validate_url("https://example.com/paper.pdf") ==
      "https://example.com/paper.pdf")
check("a normal http URL is allowed",
      fulltext._validate_url("http://example.com/paper.pdf") ==
      "http://example.com/paper.pdf")
try:
    fulltext._http_get("file:///etc/passwd")
    check("_http_get refuses file://", False, "returned bytes")
except ValueError:
    check("_http_get refuses file://", True)

check("a safe arxiv id keeps its readable cache key",
      fulltext._cache_key({"arxiv_id": "2301.12345"}) == "2301.12345")
for bad in ("../../../../tmp/foo", "/tmp/bar", "hep-th/9901001"):
    key = fulltext._cache_key({"arxiv_id": bad})
    check("an unsafe arxiv id is hashed: " + bad,
          len(key) == 16 and all(c in "0123456789abcdef" for c in key), key)
    check("the hashed key stays a single path component", "/" not in key and ".." not in key, key)

# ---- request body limits ---------------------------------------------------
class FakeHandler:
    _read_json = web.Handler._read_json

    def __init__(self, length=None, body=b"{}", transfer_encoding=None):
        self.headers = {}
        if length is not None:
            self.headers["Content-Length"] = length
        if transfer_encoding is not None:
            self.headers["Transfer-Encoding"] = transfer_encoding
        self.rfile = io.BytesIO(body)


def body_error(**kwargs):
    try:
        FakeHandler(**kwargs)._read_json()
    except web._BodyError as e:
        return e.code
    return None


check("an invalid Content-Length is a 400", body_error(length="abc") == 400)
check("a negative Content-Length is a 400", body_error(length="-1") == 400)
check("an oversized body is a 413",
      body_error(length=str(web.MAX_BODY_BYTES + 1)) == 413)
check("chunked bodies are refused", body_error(transfer_encoding="chunked") == 400)
check("a truncated body is a 400", body_error(length="50", body=b"{}") == 400)
check("a valid JSON object is returned",
      FakeHandler(length="2", body=b"{}")._read_json() == {})
check("a JSON array is not accepted as a payload",
      FakeHandler(length="2", body=b"[]")._read_json() is None)

print()
if fails:
    print("FAILED:", ", ".join(fails))
    raise SystemExit(1)
print("all checks passed")
