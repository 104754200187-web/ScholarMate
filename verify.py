"""Citation verification, two layers:
1. Rules: every [n] must map to a retrieved paper and match its title.
2. LLM: judge whether each cited paper actually supports the
   attached claim (OpenScholar-style attribution check).

Pure stdlib.
"""

import difflib
import re

import llm

REF_LINE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*$")
REF_YEAR = re.compile(r"\s*\(\d{4}\)\s*$")
SECTION_RE = re.compile(r"^##[ \t]*(.+?)[ \t]*$", re.MULTILINE)

# How much of a cited paper's abstract is shown as evidence when no full text
# could be read. The generator is shown 800 chars per paper (graph.build_context)
# while this layer was cut to 600, so a fact stated in the tail of an abstract
# (method, inclusion criteria, limitation) could reach the writer but not the
# judge that clears it.
ABSTRACT_EVIDENCE_CHARS = 1200


def split_sections(answer):
    """Split a report into (title, body) pairs on its `##` headings.

    Text before the first heading is kept as a (None, text) pair so nothing is
    dropped, and a text with no headings at all comes back as one (None, text)
    pair. The per-sentence splitter (sentences.py) and the survey rule checks
    below both read a report through this function, so "which section is this"
    means the same thing in both layers.
    """
    text = answer or ""
    marks = list(SECTION_RE.finditer(text))
    if not marks:
        return [(None, text)]
    out = []
    head = text[:marks[0].start()].strip()
    if head:
        out.append((None, head))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1).strip(), text[m.end():end].strip()))
    return out


def norm_key(text):
    """Lowercased alphanumerics plus CJK, so 「RNA基础模型」 matches 「RNA 基础模型」."""
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", (text or "").lower())


# Section titles a survey report may carry besides the chosen directions
# (SURVEY_PROMPT asks for 摘要 / 争议与空白 / 参考文献; the rest are harmless
# variants a writer reaches for).
_EXTRA_SECTIONS = (
    "摘要", "abstract", "summary", "小结", "总结", "引言", "introduction",
    "争议与空白", "空白", "gaps", "openquestions", "limitations", "局限",
    "讨论", "discussion", "结论", "conclusion", "展望", "outlook",
    "参考文献", "references", "引用文献", "文献列表",
)


def _direction_key(direction, index):
    """Stable id for a direction, so the "missing section" check can look it up."""
    return direction.get("id") if direction.get("id") not in (None, "") else index


def _match_direction(title, directions):
    """The direction a section title names, or None."""
    key = norm_key(title)
    if not key:
        return None
    best = None
    for d in directions:
        for field in ("name", "en"):
            name = norm_key(d.get(field))
            if len(name) < 4:
                continue
            if name in key or (len(key) >= 4 and key in name):
                return d
            ratio = difflib.SequenceMatcher(None, name, key).ratio()
            if ratio >= 0.6 and (best is None or ratio > best[0]):
                best = (ratio, d)
    return best[1] if best else None


def survey_items(answer, directions):
    """Rule items that only make sense for a structured survey report.

    The report's contract (see SURVEY_PROMPT) is one `##` section per chosen
    direction, every one of them carrying [n] citations. These checks catch the
    report and the plan drifting apart -- a direction that was searched but
    never written up, a section with nothing checkable in it, a section that
    belongs to no direction. None of them fired on the ten saved surveys (each
    direction had a section with >=6 distinct citations), so they are guards
    rather than fixes; they cost one pass over the text and name the section
    they complain about.
    """
    if not directions:
        return []
    items = []
    seen = set()
    for title, body in split_sections(answer):
        if not title:
            continue
        direction = _match_direction(title, directions)
        if direction is not None:
            seen.add(_direction_key(direction, directions.index(direction)))
            if not extract_citations(body):
                items.append({
                    "status": "warn",
                    "message": f"方向小节「{title}」没有任何 [n] 引用：这一节的结论无从核对",
                })
            continue
        key = norm_key(title)
        if any(extra in key for extra in _EXTRA_SECTIONS):
            continue
        items.append({
            "status": "warn",
            "message": f"报告里有目录外的小节「{title}」：它不属于本次选定的 "
                       f"{len(directions)} 个方向",
        })
    for i, d in enumerate(directions):
        if _direction_key(d, i) in seen:
            continue
        items.append({
            "status": "warn",
            "message": f"报告缺少方向「{d.get('name') or d.get('en') or _direction_key(d, i)}」"
                       f"的小节：该方向的检索结果没有写进正文",
        })
    return items


def _body(answer):
    """Return (body, references), cutting at the References heading.

    Both headings are recognised: the Q&A prompt asks for `References:` while a
    Chinese review report ends with `## 参考文献`. A heading the splitter does
    not know leaves the reference list inside the body, so every [n] looks like
    a citation that is missing from the list.
    """
    for marker in ("References:", "References\n", "**References**", "References"):
        idx = answer.find(marker)
        if idx != -1:
            return answer[:idx], answer[idx + len(marker):]
    idx = _find_cjk_heading(answer)
    if idx != -1:
        line_end = answer.find("\n", idx)
        return answer[:idx], answer[line_end + 1 if line_end != -1 else idx:]
    return answer, ""


_CJK_HEADING = re.compile(r"^[#\s>]*(参考文献|引文|文献列表)\s*[:：]?\s*$", re.M)


def _find_cjk_heading(answer):
    """Position of a Chinese references heading, or -1.

    Anchored to a whole line: 参考文献 shows up inside ordinary prose ("检索
    参考文献时…"), and cutting the body there would silently drop real claims.
    """
    m = _CJK_HEADING.search(answer)
    return m.start() if m else -1


def extract_citations(answer):
    body, _ = _body(answer)
    return {int(n) for n in re.findall(r"\[(\d+)\]", body)}


def extract_references(answer):
    _, refs = _body(answer)
    found = {}
    for line in refs.splitlines():
        m = REF_LINE.match(line)
        if m:
            title = REF_YEAR.sub("", m.group(2)).strip()
            found[int(m.group(1))] = title
    return found


def normalize(title):
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def title_matches(reference_title, paper_title, threshold=0.6):
    ref = normalize(reference_title)
    paper = normalize(paper_title)
    if not ref or not paper:
        return False
    if ref in paper or paper in ref:
        return True
    return difflib.SequenceMatcher(None, ref, paper).ratio() >= threshold


def verify(answer, papers, directions=None):
    """Return a list of (status, message) lines. Status in {"ok", "warn", "bad"}.

    `directions` (survey runs only) adds the report-shape checks of
    survey_items() to the citation checks, so a report that dropped a chosen
    direction is flagged here -- by rules, before any model is asked.
    """
    by_index = {i + 1: p["title"] for i, p in enumerate(papers)}
    cited = extract_citations(answer)
    references = extract_references(answer)
    total = len(papers)
    lines = []

    for n in sorted(cited):
        if n < 1 or n > total:
            lines.append(("bad", f"[{n}] cited in body but out of range (only {total} papers)"))
            continue
        if n not in references:
            lines.append(("bad", f"[{n}] cited in body but missing from References: {by_index[n]}"))
            continue
        if not title_matches(references[n], by_index[n]):
            lines.append((
                "bad",
                f"[{n}] reference title doesn't match retrieved paper:\n"
                f"      ref:  {references[n]}\n"
                f"      real: {by_index[n]}",
            ))
            continue
        lines.append(("ok", f"[{n}] ok -> {by_index[n]}"))

    for n in sorted(set(references) - cited):
        lines.append(("warn", f"[{n}] in References but never cited in body"))

    if not cited:
        lines.append(("bad", "no citations found in the answer body"))
    lines.extend((it["status"], it["message"]) for it in survey_items(answer, directions or []))
    return lines


def report(answer, papers, directions=None):
    lines = verify(answer, papers, directions)
    if not any(status == "bad" for status, _ in lines):
        lines.insert(0, ("ok", "all citations verified against retrieved papers"))
    return ["--- Citation verification ---"] + [
        f"{'OK ' if status == 'ok' else 'WARN' if status == 'warn' else 'BAD'} {msg}"
        for status, msg in lines
    ]


VERIFY_PROMPT = """You are verifying the factual grounding of citations in a \
literature answer. For every [n] cited in the answer body, decide whether the \
claim attached to it is actually supported by the cited paper's abstract.
Reply ONLY with JSON:
{"verdicts": [{"n": <int>, "supported": <bool>, "reason": "<short reason, max 15 words>"}]}
Rules:
1. Include every [n] that appears in the answer body.
2. supported=true only if the paper's abstract directly supports the attached \
claim; if the claim is not covered, exaggerated, or contradicts the abstract, \
set supported=false.
3. Do not be lenient: unsupported claims must be flagged.
4. reason must be short and contain no double quotes."""


def llm_check(answer, papers):
    """Attribution check: is each [n] claim supported by its paper's abstract?"""
    cited = [n for n in extract_citations(answer) if 1 <= n <= len(papers)]
    if not cited:
        return [("bad", "no citations found in the answer body")]
    lines = []
    for start in range(0, len(cited), 5):
        batch = sorted(cited)[start:start + 5]
        blocks = []
        for n in batch:
            p = papers[n - 1]
            abstract = (p["abstract"] or "")[:ABSTRACT_EVIDENCE_CHARS]
            blocks.append(f"[{n}] {p['title']} ({p['year']})\nAbstract: {abstract}")
        messages = [
            {"role": "system", "content": VERIFY_PROMPT},
            {
                "role": "user",
                "content": f"Answer:\n{answer}\n\nCited papers:\n" + "\n\n".join(blocks),
            },
        ]
        content = llm.chat(messages, max_tokens=1024, temperature=0)
        verdicts = llm.extract_verdicts(content, id_field="n")
        if not verdicts:
            lines.append(("bad", f"LLM returned no parseable verdicts for batch {batch}"))
            continue
        for v in verdicts:
            try:
                n = int(v["n"])
            except (KeyError, TypeError, ValueError):
                continue
            if n not in batch:
                continue
            judged = isinstance(v.get("supported"), bool)
            ok = bool(v.get("supported")) if judged else False
            reason = str(v.get("reason", "")).strip()
            title = papers[n - 1]["title"]
            lines.append((
                "ok" if ok else ("warn" if not judged else "bad"),
                f"[{n}] LLM: claim "
                f"{'supported' if ok else ('UNJUDGED' if not judged else 'NOT supported')}"
                f" by -> {title} | {reason}",
            ))
    return lines


def llm_report(answer, papers):
    lines = llm_check(answer, papers)
    return ["--- LLM citation verification ---"] + [
        f"{'OK ' if status == 'ok' else ('WARN' if status == 'warn' else 'BAD')} {msg}"
        for status, msg in lines
    ]


def llm_audit(answer, papers):
    """Structured attribution audit (one LLM pass, same prompts/log as
    llm_check). Returns {"lines": [...], "verdicts": [{"n", "supported",
    "reason", "title"}]}."""
    cited = [n for n in extract_citations(answer) if 1 <= n <= len(papers)]
    lines, verdicts = [], []
    if not cited:
        lines.append(("bad", "no citations found in the answer body"))
        return {"lines": lines, "verdicts": verdicts}
    for start in range(0, len(cited), 5):
        batch = sorted(cited)[start:start + 5]
        blocks = []
        for n in batch:
            p = papers[n - 1]
            abstract = (p["abstract"] or "")[:ABSTRACT_EVIDENCE_CHARS]
            blocks.append(f"[{n}] {p['title']} ({p['year']})\nAbstract: {abstract}")
        messages = [
            {"role": "system", "content": VERIFY_PROMPT},
            {
                "role": "user",
                "content": f"Answer:\n{answer}\n\nCited papers:\n" + "\n\n".join(blocks),
            },
        ]
        content = llm.chat(messages, max_tokens=1024, temperature=0)
        batch_verdicts = llm.extract_verdicts(content, id_field="n")
        if not batch_verdicts:
            lines.append(("bad", f"LLM returned no parseable verdicts for batch {batch}"))
            continue
        for v in batch_verdicts:
            try:
                n = int(v["n"])
            except (KeyError, TypeError, ValueError):
                continue
            if n not in batch:
                continue
            judged = isinstance(v.get("supported"), bool)
            ok = bool(v.get("supported")) if judged else False
            reason = str(v.get("reason", "")).strip()
            title = papers[n - 1]["title"]
            verdicts.append({"n": n, "supported": ok, "judged": judged,
                             "reason": reason, "title": title})
            lines.append((
                "ok" if ok else ("warn" if not judged else "bad"),
                f"[{n}] LLM: claim "
                f"{'supported' if ok else ('UNJUDGED' if not judged else 'NOT supported')}"
                f" by -> {title} | {reason}",
            ))
    return {"lines": lines, "verdicts": verdicts}


def retraction_items(answer, papers):
    """Rule items for citations that point at a retracted paper."""
    items = []
    for n in sorted(extract_citations(answer)):
        if not (1 <= n <= len(papers)):
            continue
        paper = papers[n - 1]
        if paper.get("retracted"):
            note = paper.get("retraction_note") or "该论文已被撤稿"
            items.append({
                "status": "bad",
                "message": f"[{n}] cites a RETRACTED paper: {paper['title']} ({note})",
            })
    return items


def audit_rules(answer, papers, directions=None):
    """Structured rule-layer audit. Returns {"lines": [...], "items":
    [{"status", "message"}]} mirroring verify()."""
    items = [{"status": s, "message": m} for s, m in verify(answer, papers, directions)]
    items.extend(retraction_items(answer, papers))
    if not any(it["status"] == "bad" for it in items):
        items.insert(0, {"status": "ok", "message": "all citations verified against retrieved papers"})
    lines = ["--- Citation verification ---"] + [
        f"{'OK ' if it['status'] == 'ok' else 'WARN' if it['status'] == 'warn' else 'BAD'} {it['message']}"
        for it in items
    ]
    return {"lines": lines, "items": items}
