"""Literature pack: the papers behind an answer as one ZIP (pure stdlib).

An answer cites 10-20 papers; a reader wants them locally, not 20 tabs. This
module fetches the openly available PDFs (arXiv / OA hosts) in parallel --
reusing fulltext's download guard rails (size cap, UA, arXiv 3s gap) -- and
packs them with the citation list and an honest per-paper report, so the
paywalled ones are named instead of silently dropped.

    python3 pdfpack.py <run_id> [out.zip]

Guard rails: at most MAX_PAPERS papers, WORKERS parallel fetches, a single
attempt per paper, and a wall-clock BUDGET_S after which the remaining papers
are reported as "not attempted" rather than holding the request open.
"""

import datetime
import io
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import exports
import fulltext
import search

MAX_PAPERS = int(os.environ.get("PDFPACK_MAX_PAPERS", "40"))
WORKERS = int(os.environ.get("PDFPACK_WORKERS", "4"))
BUDGET_S = float(os.environ.get("PDFPACK_BUDGET_S", "90"))
TIMEOUT = float(os.environ.get("PDFPACK_TIMEOUT", str(fulltext.DOWNLOAD_TIMEOUT)))
MAX_TOTAL_BYTES = int(os.environ.get("PDFPACK_MAX_TOTAL_BYTES", str(150 * 1000 * 1000)))
# Bigger than fulltext's cap on purpose: nothing is parsed here, only stored,
# and a 12 MB OA PDF (Frontiers) is a normal read, not a pathological one. The
# total cap below is what keeps a pack sane.
MAX_FILE_BYTES = int(os.environ.get("PDFPACK_MAX_BYTES", str(40 * 1000 * 1000)))
# Per paper: original URL, OpenAlex OA copy, and the PDF a landing page points
# at. Bounded so one hostile host cannot eat the whole budget.
MAX_ATTEMPTS = int(os.environ.get("PDFPACK_MAX_ATTEMPTS", "3"))

# Publishers answer 403 to a bot long before a human, so the URL a search result
# carries is often not the downloadable copy. OpenAlex ships Unpaywall's
# best open-access location per DOI, and one batch request resolves 50 papers at
# once -- cheap enough to always try before admitting defeat. That lookup lives
# in search.openalex_oa_pdf_urls (shared with the per-sentence stage) and reads
# OPENALEX_OA_BATCH / OPENALEX_OA_TIMEOUT.

# Windows/Word refuse these in a filename, and a leading/trailing dot or space
# is legal in Python but not on Windows either
_BAD_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def _clean_name(text, limit=110):
    s = _BAD_CHARS.sub(" ", text or "")
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s[:limit].strip(" .") or "untitled"


def filename_for(index, paper):
    """`07 - Some title (2024).pdf` -- numbered so the ZIP keeps reading order."""
    year = paper.get("year")
    title = _clean_name(paper.get("title") or "untitled")
    return f"{index:02d} - {title}" + (f" ({year})" if year else "") + ".pdf"


_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.I)


def _arxiv_from_doi(paper):
    """`10.48550/arXiv.2101.00001` is an arXiv preprint wearing a DOI; the PDF
    is one unauthenticated GET away, no lookup needed."""
    m = _ARXIV_DOI.match(_doi_key(paper))
    return "https://arxiv.org/pdf/" + m.group(1) if m else ""


# DOI normalisation, the landing-page link reader and the "is this a PDF?" test
# live in fulltext.py, which needs all three for every full-text download; the
# OpenAlex open-access lookup lives in search.py (it owns the API client).
_doi_key = fulltext.doi_key


def _fetch(url, max_bytes=None):
    """One attempt per URL: a batch cannot spend the budget on retries."""
    fulltext._arxiv_wait(url)
    return fulltext._http_get(url, timeout=TIMEOUT, retries=1,
                              max_bytes=max_bytes or MAX_FILE_BYTES)


class _ByteBudget:
    """Thread-safe reservation for bytes held by parallel PDF downloads.

    A worker reserves the largest possible response before it starts, then
    settles the reservation to the bytes it actually keeps. Reserving first is
    what bounds the *inflight* total too: checking only after every download has
    finished would let four 40 MB responses sit in memory before the ZIP cap is
    applied.
    """

    def __init__(self, limit):
        self.limit = max(0, int(limit))
        self.left = self.limit
        self.lock = Lock()

    def reserve(self, size):
        with self.lock:
            taken = min(max(0, int(size)), self.left)
            self.left -= taken
            return taken

    def release(self, size):
        with self.lock:
            self.left = min(self.limit, self.left + max(0, int(size)))

    def settle(self, reserved, actual):
        """Keep `actual` bytes reserved and return the unused part."""
        self.release(max(0, int(reserved) - int(actual)))


def download_papers(papers, should_cancel=None, oa=None):
    """Fetch every available PDF in parallel. Never raises; returns one dict
    per paper (`ok`, `note`, `data`, `url`, `via`).

    Each paper is tried at its own URL first and, when that fails, at the
    open-access copy OpenAlex knows about (`oa`).
    """
    oa = {} if oa is None else oa
    deadline = time.monotonic() + BUDGET_S
    budget = _ByteBudget(MAX_TOTAL_BYTES)

    def not_attempted(idx, paper, note):
        return {"index": idx, "paper": paper, "ok": False, "note": note,
                "attempted": False}

    def candidates(paper):
        urls = []
        own = fulltext.pdf_url_for(paper)
        if own:
            urls.append((own, "原始链接"))
        alt = oa.get(_doi_key(paper))
        if alt and alt not in [u for u, _ in urls]:
            urls.append((alt, "开放获取副本（OpenAlex）"))
        arx = _arxiv_from_doi(paper)
        if arx and arx not in [u for u, _ in urls]:
            urls.append((arx, "arXiv 预印本"))
        return urls

    def work(item):
        idx, paper = item
        if should_cancel and should_cancel():
            return not_attempted(idx, paper, "已取消")
        queue = candidates(paper)
        if not queue:
            return {"index": idx, "paper": paper, "ok": False,
                    "note": "没有公开 PDF 链接（需在出版商处获取）",
                    "attempted": False}
        first_url = queue[0][0]
        failures, seen, attempts, oversized = [], set(), 0, False
        while queue and attempts < MAX_ATTEMPTS:
            url, via = queue.pop(0)
            if not url or url in seen:
                continue
            seen.add(url)
            if time.monotonic() >= deadline - 1:
                note = (f"超出打包时间上限（{int(BUDGET_S)}s），未尝试下载"
                        if not failures else
                        "；".join(failures) + f"（已尝试 {attempts} 次，时间上限内未取到）")
                return not_attempted(idx, paper, note)
            attempts += 1
            reserved = budget.reserve(MAX_FILE_BYTES)
            if not reserved:
                out = not_attempted(
                    idx, paper,
                    f"超出打包合计体积上限（>{MAX_TOTAL_BYTES / 1e6:.0f} MB），未尝试下载")
                out["oversized"] = True
                return out
            try:
                data = _fetch(url, max_bytes=reserved)
            except Exception as e:
                budget.release(reserved)
                failures.append(f"{via}下载失败：{e}")
                if "too large" in str(e).lower():
                    oversized = True
                continue
            size = len(data)
            if fulltext.looks_like_pdf(data):
                if size > MAX_FILE_BYTES:
                    budget.release(reserved)
                    failures.append(f"{via}超过单篇体积上限")
                    oversized = True
                    continue
                budget.settle(reserved, size)
                return {"index": idx, "paper": paper, "ok": True, "data": data,
                        "url": url, "via": via, "attempted": True}
            # a landing page: the PDF it advertises gets tried first
            budget.release(reserved)
            inner = fulltext.pdf_link_from_html(data, url)
            if inner and inner not in seen:
                queue.insert(0, (inner, "落地页内的 PDF 链接"))
            failures.append(f"{via}返回的不是 PDF（可能是落地页）")
        out = {"index": idx, "paper": paper, "ok": False,
               "note": "；".join(failures), "url": first_url, "attempted": True}
        if oversized:
            out["oversized"] = True
        return out

    items = [(i, p) for i, p in enumerate(papers, 1)]
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(WORKERS, len(items)))) as ex:
        return list(ex.map(work, items))


def _report(papers, results, question, run_id, packed):
    """Plain-text manifest: what is in the ZIP and why the rest is not."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_bytes = sum(len(r["data"]) for r in results if r["ok"] and r["index"] in packed)
    lines = [
        "ScholarMate 文献打包报告",
        "=" * 46,
        f"生成时间：{now}",
        f"运行 ID：{run_id}",
        f"研究问题：{question or '（未记录）'}",
        f"论文 {len(papers)} 篇 · 已打包 PDF {len(packed)} 篇"
        f"（{total_bytes / 1e6:.1f} MB）· 未打包 {len(papers) - len(packed)} 篇",
        "",
        "说明：只打包可公开获取的 PDF（原始链接失败时会自动改用 OpenAlex",
        "      提供的开放获取副本）；仍失败的通常是纯付费墙文献，请用清单里的",
        "      DOI / 来源页链接自行获取。若某项显示「未尝试下载」，可稍后重新打包。",
        "",
        "已打包",
        "-" * 46,
    ]
    for r in results:
        if r["index"] in packed:
            lines.append(f"  {r['index']:>3}. {r['paper'].get('title') or ''} "
                         f"— {len(r['data']) / 1e6:.1f} MB — {r.get('url') or ''}"
                         f"（{r.get('via') or ''}）")
    lines += ["", "未打包", "-" * 46]
    missing = [r for r in results if r["index"] not in packed]
    if not missing:
        lines.append("  （无，全部论文都已打包）")
    for r in missing:
        p = r["paper"]
        where = []
        if p.get("doi"):
            where.append("DOI " + p["doi"])
        if p.get("url"):
            where.append(p["url"])
        lines.append(f"  {r['index']:>3}. {p.get('title') or ''} — {r['note']}")
        if where:
            lines.append("       " + " ｜ ".join(where))
    return "\n".join(lines) + "\n"


def build_zip(papers, store=None, question="", run_id=""):
    """Pack the downloadable PDFs + manifests. Returns (zip_bytes, summary)."""
    papers = list(papers)[:MAX_PAPERS]
    oa = search.openalex_oa_pdf_urls(papers)
    results = download_papers(papers, oa=oa)
    packed, total = set(), 0
    oversized = [r["index"] for r in results if r.get("oversized")]
    for r in results:
        if not r["ok"]:
            continue
        size = len(r["data"])
        if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES:
            if r["index"] not in oversized:
                oversized.append(r["index"])
            r["note"] = (f"超出打包体积上限（单篇 >{MAX_FILE_BYTES / 1e6:.0f} MB"
                         f" 或合计 >{MAX_TOTAL_BYTES / 1e6:.0f} MB），未打包，可用链接单独下载")
            continue
        packed.add(r["index"])
        total += size

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.csv", exports.papers_csv(papers, store).encode("utf-8"))
        z.writestr("manifest.bib", exports.bibtex(papers).encode("utf-8"))
        z.writestr("download-report.txt",
                   _report(papers, results, question, run_id, packed).encode("utf-8"))
        for r in results:
            if r["index"] in packed:
                z.writestr("papers/" + filename_for(r["index"], r["paper"]),
                           r["data"], compress_type=zipfile.ZIP_STORED)

    failure_reasons = {}
    for r in results:
        if r["index"] in packed:
            continue
        reason = r["note"]
        key = ("付费墙 / 无公开 PDF" if "下载失败" in reason or "没有公开 PDF" in reason
               else reason.split("：")[0])
        failure_reasons[key] = failure_reasons.get(key, 0) + 1
    summary = {
        "total": len(papers),
        "pdfs": len(packed),
        "bytes": total,
        "missing": len(papers) - len(packed),
        "not_attempted": len([r for r in results if not r.get("attempted")]),
        "oversized": oversized,
        "reasons": failure_reasons,
        "oa_candidates": len(oa),
    }
    return buf.getvalue(), summary


def main(argv=None):
    import sys
    import json
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    run_id = argv[0]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import screening
    import web as _web
    job = _web._load_job_or_snapshot(run_id)
    if not job:
        print("unknown run id:", run_id)
        return 1
    data, summary = build_zip(job["papers"], screening.load(run_id),
                              job.get("question", ""), run_id)
    out = argv[1] if len(argv) > 1 else f"/tmp/scholarpack-{run_id}.zip"
    with open(out, "wb") as f:
        f.write(data)
    print(json.dumps(summary, ensure_ascii=False))
    print(f"wrote {out} ({len(data) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
