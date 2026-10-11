"""Citation export helpers (pure stdlib).

Bibliographies (BibTeX / RIS / GB/T 7714 / CSL-JSON) plus the screening ledger
exports: a PRISMA-style CSV/Markdown table that records, per paper, the 纳入 /
排除 / 待定 decision, the reason and the note, so a review's selection step can
be reproduced and reported.
"""

import json
import re
import time

import screening


def _year(p):
    y = p.get("year")
    if isinstance(y, int):
        return y
    if y is not None and str(y).isdigit():
        return int(y)
    return None


def _clean(value):
    text = re.sub(r"[\s\n]+", " ", str(value or "")).strip()
    return text.replace("{", "").replace("}", "")


def _key(p, idx):
    surname = _authors_for_key(p) or "anon"
    year = _year(p) or "nd"
    stem = re.sub(r"[^a-zA-Z0-9]", "", surname)[:20] or "paper"
    word = re.sub(r"[^a-zA-Z0-9]", "", re.split(r"\s", (p.get("title") or ""))[0])[:15]
    return f"{stem}{year}{word or idx}".lower()


def _authors_for_key(p):
    authors = p.get("authors") or []
    if not authors:
        return ""
    name = authors[0].strip()
    if "," in name:
        return name.split(",")[0].strip()
    parts = name.split()
    return parts[-1] if parts else name


def _format_author(name):
    name = _clean(name)
    if "," in name:
        return name
    parts = name.split()
    if len(parts) <= 1:
        return name
    given, family = " ".join(parts[:-1]), parts[-1]
    initials = " ".join(f"{w[0]}." for w in re.split(r"[\s\-]+", given) if w)
    return f"{family}, {initials}".strip() if initials else f"{family}, {given}"


def _bibtex_entry(p, idx):
    authors = ", ".join(_format_author(a) for a in (p.get("authors") or [])[:12]) or "{Anonymous}"
    fields = [
        f"  title = {{{_clean(p.get('title'))}}}",
        f"  author = {{{authors}}}",
    ]
    if _year(p):
        fields.append(f"  year = {{{_year(p)}}}")
    if p.get("venue"):
        fields.append(f"  journal = {{{_clean(p.get('venue'))}}}")
    fields.append(f"  url = {{{_clean(p.get('url'))}}}" if p.get("url") else "  url = {}")
    if p.get("doi"):
        fields.append(f"  doi = {{{_clean(p.get('doi'))}}}")
    if p.get("arxiv_id"):
        fields.append(f"  eprint = {{{_clean(p.get('arxiv_id'))}}}")
    if p.get("pdf_url"):
        fields.append(f"  annote = {{PDF: {_clean(p.get('pdf_url'))}}}")
    fields.append(f"  note = {{retrieved via ScholarMate from {p.get('source', '')}}}")
    entry_type = "article" if (_year(p) or not p.get("arxiv_id")) else "misc"
    body = ",\n".join(fields)
    return f"@{entry_type}{{{_key(p, idx)},\n{body}\n}}\n"


def bibtex(papers):
    return "\n".join(_bibtex_entry(p, i) for i, p in enumerate(papers, 1))


def _ris_entry(p):
    lines = ["TY  - JOUR"]
    for a in (p.get("authors") or [])[:12]:
        lines.append(f"AU  - {_clean(a)}")
    if not (p.get("authors") or []):
        lines.append("AU  - Anonymous")
    lines.append(f"TI  - {_clean(p.get('title'))}")
    if p.get("venue"):
        lines.append(f"JO  - {_clean(p.get('venue'))}")
    if _year(p):
        lines.append(f"PY  - {_year(p)}")
    if p.get("doi"):
        lines.append(f"DO  - {_clean(p.get('doi'))}")
    if p.get("url"):
        lines.append(f"UR  - {_clean(p.get('url'))}")
    if p.get("abstract"):
        lines.append(f"AB  - {_clean(p.get('abstract'))[:500]}")
    lines.append(f"DB  - ScholarMate ({p.get('source', '')})")
    lines.append("ER  - ")
    return "\n".join(lines)


def ris(papers):
    return "\n\n".join(_ris_entry(p) for p in papers)


def gbt7714(papers):
    out = []
    for i, p in enumerate(papers, 1):
        authors = p.get("authors") or []
        head = ", ".join(authors[:3])
        if len(authors) > 3:
            head += ", 等"
        head = head or "佚名"
        year = f"{_year(p)}." if _year(p) else ""
        title = _clean(p.get("title"))
        link = f"[EB/OL]. {_clean(p.get('url'))}." if p.get("url") else ""
        out.append(f"[{i}] {head}. {title}[J]. {year}{link}")
    return "\n".join(out)


def csl_json(papers):
    items = []
    for p in papers:
        issued = None
        if _year(p):
            issued = {"date-parts": [[_year(p)]]}
        item = {
            "id": p.get("doi") or p.get("arxiv_id") or _key(p, len(items) + 1),
            "type": "article-journal",
            "title": _clean(p.get("title")),
            "author": [{"literal": _clean(a)} for a in (p.get("authors") or [])]
            or [{"literal": "Anonymous"}],
        }
        if issued:
            item["issued"] = issued
        if p.get("venue"):
            item["container-title"] = _clean(p.get("venue"))
        if p.get("doi"):
            item["DOI"] = p.get("doi")
        if p.get("url"):
            item["URL"] = p.get("url")
        if p.get("abstract"):
            item["abstract"] = _clean(p.get("abstract"))[:500]
        items.append(item)
    return json.dumps(items, ensure_ascii=False, indent=1)


def per_paper_bibtex(papers):
    return {str(i): _bibtex_entry(p, i) for i, p in enumerate(papers, 1)}


# ------------------------------------------------------- screening ledger

def _decision_cells(paper, store):
    """(decision label, reason label, note, updated_at) for one paper."""
    rec = (store or {}).get(screening.paper_key(paper)) or {}
    decision = rec.get("decision") or ""
    return (
        screening.DECISIONS.get(decision, "未筛选"),
        screening.reason_label(rec.get("reason") or "") if decision else "",
        rec.get("note") or "",
        rec.get("updated_at") or "",
    )


def _decision_counts(papers, store):
    counts = {d: 0 for d in screening.DECISIONS}
    for p in papers:
        decision = ((store or {}).get(screening.paper_key(p)) or {}).get("decision")
        if decision in counts:
            counts[decision] += 1
    return counts


def _csv_cell(value):
    text = re.sub(r"[\r\n]+", " ", str(value if value is not None else "")).strip()
    return '"' + text.replace('"', '""') + '"' if any(c in text for c in ',"') else text


def prisma_csv(papers, store):
    """Screening ledger as CSV (UTF-8 BOM so Excel opens it correctly)."""
    rows = [["序号", "标题", "年份", "被引", "来源", "DOI", "筛选决定", "理由", "备注", "记录时间"]]
    for i, p in enumerate(papers, 1):
        decision, reason, note, updated = _decision_cells(p, store)
        rows.append([
            str(i), _clean(p.get("title")), _year(p) or "",
            p.get("citations") if p.get("citations") is not None else 0,
            p.get("source") or "", p.get("doi") or "",
            decision, reason, note, updated,
        ])
    lines = [",".join(_csv_cell(c) for c in row) for row in rows]
    return "\ufeff" + "\n".join(lines) + "\n"


def papers_csv(papers, store=None, include_abstract=True):
    """The retrieved list as one CSV row per paper (Excel opens it directly).

    This is the sheet people keep next to a manuscript: authors, venue, year,
    citations, DOI/links, retraction flag and -- when a screening ledger is
    passed in -- the 纳入/排除/待定 decision with its reason. The abstract stays
    in because a list without abstracts is unusable offline.
    """
    head = ["序号", "标题", "作者", "年份", "刊物/会议", "被引", "来源", "DOI", "链接",
            "PDF", "已撤稿", "筛选决定", "筛选理由"]
    if include_abstract:
        head.append("摘要")
    rows = [head]
    for i, p in enumerate(papers, 1):
        decision, reason, _note, _updated = _decision_cells(p, store)
        row = [
            str(i), _clean(p.get("title")), _clean("; ".join(p.get("authors") or [])),
            str(_year(p) or ""), _clean(p.get("venue")), str(p.get("citations") or 0),
            p.get("source") or "", p.get("doi") or "", p.get("url") or "",
            p.get("pdf_url") or "", "是" if p.get("retracted") else "",
            decision, reason,
        ]
        if include_abstract:
            row.append(_clean(p.get("abstract")))
        rows.append(row)
    lines = [",".join(_csv_cell(c) for c in row) for row in rows]
    return "\ufeff" + "\n".join(lines) + "\n"


def evidence_csv(papers, dimensions=None, weights=None):
    """Per-paper evidence appraisal: total, grade and every dimension score.

    `dimensions` is scoring.DIMENSIONS-shaped (passed in by the caller so this
    module stays independent of the scoring module's internals).
    """
    dims = dimensions or [{"key": k, "label": l} for k, l in
                          (("relevance", "相关性"), ("directness", "直接性"),
                           ("method", "方法可靠性"), ("replication", "重复验证"),
                           ("structural", "结构支持"), ("data", "数据完整性"),
                           ("species", "物种匹配"), ("impact", "影响力"),
                           ("bias", "偏倚风险"), ("contradiction", "矛盾证据"))]
    head = ["序号", "标题", "年份", "来源", "证据总分", "等级"]
    head += [d["label"] for d in dims]
    head += ["冲突论文数", "评分说明"]
    rows = [head]
    for i, p in enumerate(papers, 1):
        ev = p.get("evidence") or {}
        scores = ev.get("dims") or {}
        row = [str(i), _clean(p.get("title")), str(_year(p) or ""), _clean(p.get("source")),
               "" if ev.get("total") is None else str(ev.get("total")), ev.get("grade") or "未评分"]
        row += ["" if scores.get(d["key"]) is None else str(scores[d["key"]]) for d in dims]
        row += [str(len(ev.get("contradicted_by") or [])), ev.get("note") or ""]
        rows.append(row)
    lines = [",".join(_csv_cell(c) for c in row) for row in rows]
    if weights:
        lines.append("")
        lines.append(_csv_cell("权重") + "," + ",".join(
            _csv_cell(f"{d['label']}={weights.get(d['key'], 0)}") for d in dims))
    return "\ufeff" + "\n".join(lines) + "\n"


def prisma_markdown(papers, store, question="", run_id=""):
    """Same ledger as Markdown: a summary block plus the per-paper table, ready
    to paste into a manuscript's study-selection section."""
    counts = _decision_counts(papers, store)
    undecided = len(papers) - sum(counts.values())
    lines = ["# 文献筛选记录（PRISMA 式）", ""]
    if question:
        lines.append(f"- 研究问题：{question}")
    if run_id:
        lines.append(f"- 检索批次：`{run_id}`")
    lines += [
        f"- 检索命中：{len(papers)} 篇",
        "- 筛选结果：纳入 {} 篇 ｜ 排除 {} 篇 ｜ 待定 {} 篇 ｜ 未筛选 {} 篇".format(
            counts["include"], counts["exclude"], counts["maybe"], undecided),
        f"- 导出时间：{time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "| # | 标题 | 年份 | 被引 | 来源 | 决定 | 理由 | 备注 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, p in enumerate(papers, 1):
        decision, reason, note, _ = _decision_cells(p, store)
        cells = [
            str(i), _clean(p.get("title")), str(_year(p) or ""),
            str(p.get("citations") if p.get("citations") is not None else 0),
            _clean(p.get("source")), decision, reason, note,
        ]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    excluded = [p for p in papers
                if ((store or {}).get(screening.paper_key(p)) or {}).get("decision") == "exclude"]
    if excluded:
        lines += ["", f"## 排除文献及理由（{len(excluded)} 篇）", ""]
        for p in excluded:
            _, reason, note, _ = _decision_cells(p, store)
            detail = "；".join(x for x in (reason, note) if x)
            lines.append(f"- {_clean(p.get('title'))}（{_year(p) or '年份未知'}）—— {detail or '未填写理由'}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    demo = [{
        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "abstract": "Survey of RAG.", "year": 2023, "citations": 100,
        "url": "https://arxiv.org/abs/2312.10997", "authors": ["Yunfan Gao", "Yun Xiong"],
        "source": "semantic_scholar", "pdf_url": "https://arxiv.org/pdf/2312.10997",
        "arxiv_id": "2312.10997", "doi": "", "seen_sources": ["semantic_scholar"],
    }]
    print(bibtex(demo))
    print("---- RIS ----")
    print(ris(demo))
    print("---- GB/T 7714 ----")
    print(gbt7714(demo))
    print("---- CSL ----")
    print(csl_json(demo))
