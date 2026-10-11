"""Citation verification, two layers:
1. Rules: every [n] must map to a retrieved paper and match its title.
2. LLM (DeepSeek): judge whether each cited paper actually supports the
   attached claim (OpenScholar-style attribution check).

Pure stdlib.
"""

import difflib
import re

import llm

REF_LINE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*$")
REF_YEAR = re.compile(r"\s*\(\d{4}\)\s*$")


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


def verify(answer, papers):
    """Return a list of (status, message) lines. Status in {"ok", "warn", "bad"}."""
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
    return lines


def report(answer, papers):
    lines = verify(answer, papers)
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
            abstract = (p["abstract"] or "")[:600]
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
            ok = bool(v.get("supported"))
            reason = str(v.get("reason", "")).strip()
            title = papers[n - 1]["title"]
            lines.append((
                "ok" if ok else "bad",
                f"[{n}] LLM: claim {'supported' if ok else 'NOT supported'} by -> {title} | {reason}",
            ))
    return lines


def llm_report(answer, papers):
    lines = llm_check(answer, papers)
    return ["--- LLM citation verification (DeepSeek) ---"] + [
        f"{'OK ' if status == 'ok' else 'BAD'} {msg}" for status, msg in lines
    ]


def llm_audit(answer, papers):
    """Structured attribution audit (one DeepSeek pass, same prompts/log as
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
            abstract = (p["abstract"] or "")[:600]
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
            ok = bool(v.get("supported"))
            reason = str(v.get("reason", "")).strip()
            title = papers[n - 1]["title"]
            verdicts.append({"n": n, "supported": ok, "reason": reason, "title": title})
            lines.append((
                "ok" if ok else "bad",
                f"[{n}] LLM: claim {'supported' if ok else 'NOT supported'} by -> {title} | {reason}",
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


def audit_rules(answer, papers):
    """Structured rule-layer audit. Returns {"lines": [...], "items":
    [{"status", "message"}]} mirroring verify()."""
    items = [{"status": s, "message": m} for s, m in verify(answer, papers)]
    items.extend(retraction_items(answer, papers))
    if not any(it["status"] == "bad" for it in items):
        items.insert(0, {"status": "ok", "message": "all citations verified against retrieved papers"})
    lines = ["--- Citation verification ---"] + [
        f"{'OK ' if it['status'] == 'ok' else 'WARN' if it['status'] == 'warn' else 'BAD'} {it['message']}"
        for it in items
    ]
    return {"lines": lines, "items": items}
