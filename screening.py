"""Screening decisions with an audit trail (PRISMA-style record keeping).

A literature review is defined as much by what it excludes as by what it keeps,
and that decision has to be reproducible months later. Every retrieved paper can
be marked 纳入 / 排除 / 待定 with a reason and a free-text note; the record is
stored per run under outputs/screening/<run_id>.json and keyed by DOI (falling
back to the normalized title when there is no DOI), so a decision survives a
re-run of the same question and feeds the exports without touching the snapshot.

Pure stdlib, no third-party dependency.
"""

import atomicio
import json
import os
import re
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENING_DIR = os.environ.get("SCREENING_DIR") or os.path.join(BASE_DIR, "outputs", "screening")

# Decision keys are stable and language-neutral: they are what gets stored in
# the JSON ledger and exported in the PRISMA table, so a label can be reworded
# later without invalidating records that were already made.
DECISIONS = {
    "include": "纳入",
    "exclude": "排除",
    "maybe": "待定",
}

# (key, label) pairs offered per decision in the UI and stored verbatim.
REASONS = {
    "include": [
        ("core_evidence", "核心证据，直接回答研究问题"),
        ("method_reference", "方法/技术可借鉴"),
        ("background", "综述或背景材料"),
        ("dataset_benchmark", "数据集或评测基准"),
        ("contrast", "对立观点，需在讨论中引用"),
    ],
    "exclude": [
        ("irrelevant", "主题不相关"),
        ("not_peer_reviewed", "非同行评审（预印本/报告/灰色文献）"),
        ("wrong_method", "研究对象或方法不适用"),
        ("duplicate", "重复文献"),
        ("retracted", "已撤稿"),
        ("no_fulltext", "无法获取全文"),
        ("language", "语言无法处理"),
        ("low_quality", "证据质量不足"),
    ],
    "maybe": [
        ("need_fulltext", "需读全文再定"),
        ("indirect", "证据间接"),
        ("unclear_quality", "质量存疑"),
        ("conflict", "与其他证据冲突"),
    ],
}

_lock = threading.Lock()


def reason_label(key):
    """Chinese label for a reason key ("" when unknown)."""
    for options in REASONS.values():
        for k, label in options:
            if k == key:
                return label
    return ""


def default_reason(decision):
    options = REASONS.get(decision) or []
    return options[0][0] if options else ""


def reasons_payload():
    """REASONS as JSON-friendly lists, for the frontend."""
    return {d: [{"key": k, "label": label} for k, label in opts]
            for d, opts in REASONS.items()}


def paper_key(paper):
    """Stable identity of a paper across runs: DOI when there is one, else the
    same normalized title the pool dedupes on."""
    doi = (paper.get("doi") or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    if doi:
        return "doi:" + doi
    title = re.sub(r"[^a-z0-9]+", " ", (paper.get("title") or "").lower()).strip()
    return "t:" + title


def _path(run_id):
    safe = re.sub(r"[^\w.\-]", "", str(run_id or ""))
    if safe in ("", ".", ".."):
        return ""
    path = os.path.join(SCREENING_DIR, safe + ".json")
    # Defense in depth: the web layer already rejects malformed run ids, but a
    # caller reaching this module directly must not be able to name a path
    # outside the screening directory.
    if os.path.realpath(os.path.dirname(path)) != os.path.realpath(SCREENING_DIR):
        return ""
    return path


def load(run_id):
    """Ledger for a run: {paper_key: {decision, reason, note, ...}}."""
    path = _path(run_id)
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            store = json.load(f)
    except (OSError, ValueError):
        return {}
    return store if isinstance(store, dict) else {}


def save(run_id, store):
    path = _path(run_id)
    if not path:
        raise ValueError("bad run id")
    os.makedirs(SCREENING_DIR, exist_ok=True)
    atomicio.write_json(path, store, indent=1)
    return path


def record(run_id, key, decision, reason="", note="", title=""):
    """Set one decision (empty `decision` clears it) and return the ledger
    summary. Reason defaults to the first option of the decision, so clicking
    one button already yields a complete, exportable record."""
    if decision not in DECISIONS:
        decision = ""
    if decision and not reason:
        reason = default_reason(decision)
    with _lock:
        store = load(run_id)
        if decision:
            store[key] = {
                "decision": decision,
                "reason": reason,
                "note": (note or "").strip(),
                "title": title or "",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            store.pop(key, None)
        save(run_id, store)
    return summary(store)


def summary(store):
    counts = {d: 0 for d in DECISIONS}
    for rec in (store or {}).values():
        decision = (rec or {}).get("decision")
        if decision in counts:
            counts[decision] += 1
    return {"counts": counts, "labels": DECISIONS, "screened": sum(counts.values())}


def payload(run_id):
    """Everything the UI needs for one run in a single round-trip."""
    store = load(run_id)
    return {
        "run": run_id,
        "decisions": store,
        "summary": summary(store),
        "reasons": reasons_payload(),
    }


def filter_papers(papers, store, only=None):
    """Subset of `papers` matching an export filter.

    only: "include" | "exclude" | "maybe" (that decision) | "undecided" (no
    record) | "screened" (any record) | falsy (no filter).
    """
    store = store or {}
    if not only:
        return list(papers)
    out = []
    for p in papers:
        rec = store.get(paper_key(p)) or {}
        decision = rec.get("decision") or ""
        if only == "undecided":
            keep = not decision
        elif only == "screened":
            keep = bool(decision)
        else:
            keep = decision == only
        if keep:
            out.append(p)
    return out


if __name__ == "__main__":
    demo = {"title": "Retrieval-Augmented Generation for Large Language Models",
            "doi": "10.1234/abc"}
    print(paper_key(demo), paper_key({"title": "No DOI Here"}))
    record("selftest", paper_key(demo), "include", note="核心综述")
    print(json.dumps(payload("selftest"), ensure_ascii=False, indent=1)[:400])
    print(json.dumps(summary(load("selftest")), ensure_ascii=False))
    record("selftest", paper_key(demo), "")
    print("cleared:", load("selftest"))
