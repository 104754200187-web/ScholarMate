"""Per-sentence evidence check (OpenScholar-style).

Split the generated answer into factual claims, retrieve supporting evidence
for each claim independently, then judge whether each claim is supported by
the (expanded) paper pool. Pure stdlib.

Numbering discipline: every [n] handled here is the answer's numbering -- the
order of the list the generator was shown. Extra papers found by the per-claim
evidence search are appended and marked ``in_answer=False``; they never shift
the meaning of an [n] that appears in the answer text.
"""

import os
import re
import time
import contextvars
from concurrent.futures import ThreadPoolExecutor, as_completed

import fulltext
import llm
import rerank
import search
import verify

# How much of a long answer gets sentence-level verification. An ask-mode answer
# is prose, so one pass over it with a low cap is right; a survey is 4-8
# sections and 25-46 cited sentences, and flattening that into 8 merged claims
# (measured on ten saved surveys: 8 verdicts against 25-46 cited sentences)
# leaves most of the report unchecked and mixes three papers into one verdict.
# Sections therefore get their own pass and their own cap, and the total is
# capped so a pathological outline cannot explode the stage.
MAX_CLAIMS = 8              # a sectionless answer (ask mode) keeps the old cap
MAX_CLAIMS_PER_SECTION = 4
MAX_CLAIMS_TOTAL = 24
SPLIT_WORKERS = int(os.environ.get("CLAIM_SPLIT_WORKERS", "4"))
# How many cited papers may be fetched for full-text evidence in one run.
# "auto" keeps one first source per claim plus four upgrades, bounded to 24;
# a positive integer is an explicit cap; 0 fetches none and -1 fetches every
# cited paper with an address (the pre-budget behaviour). Without this cap the
# prefetch queue is the citation order, so a slow 403 on paper [1] can consume
# the whole FULLTEXT_DEADLINE_S before a claim whose only support is [17] gets
# a turn.
FULLTEXT_BUDGET = os.environ.get("CLAIM_FULLTEXT_BUDGET", "auto").strip().lower()
# full-text reasons that mean "we tried and did not get the text", as opposed to
# "this paper has no open full text at all" (see fulltext.get_fulltext)
_FETCH_FAILURES = ("download failed", "timed out", "text extraction failed",
                   "extraction busy", "deadline reached", "unavailable in a recent run",
                   # a landing page a publisher handed a bot instead of the PDF:
                   # the download happened and yielded no readable paper, so the
                   # verdict rests on absent evidence exactly like a 403 does
                   "landing page")
EXTRA_EVIDENCE_BUDGET = 4  # global cap on full-text fetches for supporting-but-uncited papers
EXTRA_PER_CLAIM = 1        # at most one such fetch per claim (enough to prove full-text support)
# The per-sentence stage is dominated by LLM calls (passage rerank per cited
# paper, then the batch judgement). They are independent, so run them in
# parallel; the model provider tolerates this comfortably at 4-5 in flight.
EVIDENCE_WORKERS = int(os.environ.get("CLAIM_EVIDENCE_WORKERS", "4"))
# Supplementary evidence search. Every claim is searched (it is what finds the
# field's anchor papers -- measured: 43 papers the pool did not have, 17 of them
# with >=100 citations); the budget is only a safety valve and defaults to the
# claim cap, since the queries run concurrently and cost little extra.
# Sources: the four key-free ones that answer in 1-7s. Google Scholar is left
# out on purpose (paid quota, ~100/month, and it added 4 papers here), arXiv
# too (its CDN answers 406 here; its content arrives via S2/Crossref).
EVIDENCE_QUERY_BUDGET = int(os.environ.get("CLAIM_EVIDENCE_QUERY_BUDGET", "8"))
EVIDENCE_SOURCES = tuple(
    s.strip() for s in os.environ.get(
        "CLAIM_EVIDENCE_SOURCES",
        "openalex,semantic_scholar,europe_pmc,crossref").split(",") if s.strip()
)
JUDGE_WORKERS = int(os.environ.get("CLAIM_JUDGE_WORKERS", "3"))
EVIDENCE_PER_CLAIM = 5

SPLIT_PROMPT = """Split the following literature text into its factual claims. \
Each claim is a single statement asserting something about the literature \
(usually a sentence or clause, keeping any citation [n] exactly as written). \
Reply ONLY with JSON:
{"claims": [{"id": 1, "text": "...", "citations": [n, ...]}]}
Rules: at most {cap} claims; pick the claims that carry the substance (a result, \
a method, a comparison, a limitation) rather than the filler; keep exact wording \
-- do not rewrite, translate, merge or summarise."""


def _split_prompt(cap):
    """SPLIT_PROMPT with its cap filled in. str.format() cannot be used on it:
    the JSON skeleton in the prompt is full of braces."""
    return SPLIT_PROMPT.replace("{cap}", str(cap))


JUDGE_PROMPT = """You are checking whether each claim in an answer is supported \
by the provided papers. A claim is supported only if at least one paper's \
abstract directly backs it. Reply ONLY with JSON:
{"verdicts": [{"id": 1, "supported": true/false, "supporting": [n, ...], "reason": "<short reason>"}]}
Rules: supporting lists the paper numbers that back the claim; if none, \
supported=false and supporting=[]. reason must be short and contain no double \
quotes."""


CLAIM_FALLBACK_RE = re.compile(
    r'"id"\s*:\s*(\d+)\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"'
    r'\s*(?:,\s*"citations"\s*:\s*(\[[^\]]*\]))?'
)


def _extract_claims_fallback(content):
    claims = []
    for m in CLAIM_FALLBACK_RE.finditer(content):
        text = m.group(2).replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")
        if not text.strip():
            continue
        citations = [int(x) for x in re.findall(r"\d+", m.group(3))] if m.group(3) else []
        claims.append({"id": int(m.group(1)), "text": text, "citations": citations})
    return claims


def _citation_numbers(value):
    out = []
    for n in value or []:
        try:
            out.append(int(n))
        except (TypeError, ValueError):
            continue
    return out


def _split_one(text, cap):
    """One splitter call over `text` -> [{"text", "citations"}].

    Raises when the model cannot be reached at all, so the caller can choose
    between a retry and the local fallback.
    """
    content = llm.chat(
        [
            {"role": "system", "content": _split_prompt(cap)},
            {"role": "user", "content": text},
        ],
        max_tokens=1024,
        temperature=0,
    )
    try:
        result = llm.parse_json(content)
    except ValueError:
        result = None
    if not isinstance(result, dict):
        result = {"claims": _extract_claims_fallback(content)}
    out = []
    for c in result.get("claims", []):
        claim = str(c.get("text", "")).strip()
        if claim:
            out.append({"text": claim, "citations": _citation_numbers(c.get("citations"))})
    return out[:cap]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；;!?])\s*|\n+")


def _local_claims(text, cap):
    """Last-resort split: the cited sentences of `text`, verbatim.

    Only reached when the splitter call fails, so that one HTTP error costs
    resolution (a whole sentence instead of the substance clause inside it)
    instead of zeroing out the whole layer. A sentence with no [n] has nothing
    to be checked against, so it is dropped.
    """
    out = []
    for raw in _SENTENCE_SPLIT_RE.split(text or ""):
        sentence = re.sub(r"[ \t]+", " ", raw).strip(" \t*-#>•")
        if len(sentence) < 12 or not re.search(r"\[\d+\]", sentence):
            continue
        out.append({"text": sentence,
                    "citations": [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]})
        if len(out) >= cap:
            break
    return out


def _with_citations(claims):
    """Keep the claims that point at a paper.

    The per-sentence layer exists to check citations, and a statement carrying
    no [n] has nothing to be checked against: judged against the paper list it
    would come back 不支持 and report the report's own framing ("本次检索未覆盖
    湿实验验证…") as an unsupported finding. The numbers are read off the claim
    text, which is what the answer actually says, and the model's own list is
    only a fallback.
    """
    out = []
    for c in claims:
        numbers = [int(n) for n in re.findall(r"\[(\d+)\]", c["text"])]
        if not numbers:
            numbers = c.get("citations") or []
        if not numbers:
            continue
        out.append({"text": c["text"], "citations": numbers})
    return out


def _split_claims(text, cap):
    """Split one block of text: one call, one retry, then the local fallback."""
    for attempt in (1, 2):
        _check_cancel()
        try:
            claims = _with_citations(_split_one(text, cap))
        except Exception as e:
            _check_cancel()      # re-raises when the failure *is* the stop request
            print(f"      [warn] claim split failed ({type(e).__name__}: {e})")
            claims = []
        if claims:
            return claims
        if attempt == 1:
            time.sleep(1.5)
    claims = _local_claims(text, cap)
    print(f"      [warn] splitter unavailable for a section; fell back to "
          f"{len(claims)} of its cited sentences")
    return claims


def split_claims(answer):
    """Claims of an unstructured answer: one call, the flat cap."""
    return [dict(c, id=n) for n, c in enumerate(_split_claims(answer, MAX_CLAIMS), 1)]


_BIB_SECTIONS = ("references", "reference", "参考文献", "引用文献", "文献列表",
                 "literaturecited", "bibliography")


def _split_targets(answer):
    """Sections worth their own splitter call: prose that cites something.

    A bibliography is `[n] Title (Year)` lines with nothing to verify and a
    section without a single [n] has nothing that a paper could be checked
    against, so neither is split (the rule layer still complains about a
    direction section that cites nothing -- see verify.survey_items).
    """
    out = []
    for title, body in verify.split_sections(answer):
        if not body or not re.search(r"\[\d+\]", body):
            continue
        if title and any(bib in verify.norm_key(title) for bib in _BIB_SECTIONS):
            continue
        out.append((title, body))
    return out


def split_report_claims(answer):
    """Claims of a report, one splitter call per section.

    Measured on the ten saved surveys: the flat pass returned 8 claims for the
    25-46 cited sentences a report carries -- 17%-32% of the report checked --
    and merged statements from different sections into a single verdict. Here
    every section gets its own call and its own cap, and the section title
    travels with each claim, so a verdict can be read next to the part of the
    report it belongs to.

    The per-section cap is derived from the report's own length, so the total
    stays under MAX_CLAIMS_TOTAL without dropping a section: 8 sections get 3
    claims each, 4 sections get 4 each. Section calls are independent model
    calls, so they run concurrently. An answer with no headings at all (ask
    mode) keeps the flat path above.
    """
    sections = _split_targets(answer)
    if not sections or (len(sections) == 1 and not sections[0][0]):
        # one block of text with no heading above it: the flat pass, one call,
        # the old cap (MAX_CLAIMS)
        return split_claims(answer)
    per_section = max(1, min(MAX_CLAIMS_PER_SECTION, MAX_CLAIMS_TOTAL // len(sections)))
    results = [None] * len(sections)
    workers = max(1, min(SPLIT_WORKERS, len(sections)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(contextvars.copy_context().run, _split_claims, body, per_section): i
            for i, (_title, body) in enumerate(sections)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    claims = []
    for (title, _body), got in zip(sections, results):
        for c in got or []:
            claims.append({"id": len(claims) + 1, "text": c["text"],
                           "citations": c["citations"], "section": title})
    print(f"      {len(sections)} section(s) -> {len(claims)} claims "
          f"(<= {per_section} per section)")
    return claims


def _query_from_claim(text):
    query = re.sub(r"\[\d+\]", "", text).strip()
    return query[:120]


def _pool_support(claim_text, pool, exclude=None):
    """Best BM25 score of any pooled paper against one claim.

    Used to *order* the evidence-query and full-text queues: when a budget
    forces a cut, the claims the pool already covers best are the ones to drop.
    It is deliberately not a skip gate -- measured, dropping covered claims
    saved a few seconds and lost ~40 papers, because a claim's own paper is not
    the only relevant one.

    ``exclude`` drops titles already named by the claim. For the full-text
    budget this matters: the cited paper itself is normally in the retrieval
    pool, and counting it as independent coverage would mark every claim as
    well covered and defer the full text that is the only evidence it has.
    """
    skip = {verify.normalize(t) for t in (exclude or set()) if t}
    papers = [p for p in (pool or {}).values()
              if p.get("title") and verify.normalize(p["title"]) not in skip]
    if not papers:
        return 0.0
    docs = [fulltext.tokens((p.get("title") or "") + " " + (p.get("abstract") or ""))
            for p in papers]
    scores = fulltext.bm25_scores(claim_text, docs)
    return max(scores) if scores else 0.0


def _fulltext_budget_limit(claim_count, fetchable_count):
    """Resolve CLAIM_FULLTEXT_BUDGET against the papers that can be fetched."""
    raw = str(FULLTEXT_BUDGET).strip().lower()
    if raw in ("", "auto"):
        limit = min(24, max(8, claim_count + 4))
    else:
        try:
            limit = int(raw)
        except ValueError:
            limit = min(24, max(8, claim_count + 4))
        if limit < 0:
            limit = fetchable_count
    return min(fetchable_count, max(0, limit))


def _fetchable(paper):
    return bool(fulltext.pdf_url_for(paper) or fulltext.alternative_pdf_url(paper))


def plan_fulltext_budget(claims, papers, pool=None):
    """Choose which cited papers get a full-text download attempt.

    The old prefetch queued every cited paper in bibliography order. With four
    workers and a wall-clock deadline that is effectively a lottery: a slow
    publisher at [1] can consume the budget before [17] -- the only source for
    a later claim -- starts. Here the unit of value is a claim, not a paper:

    * A claim whose cited paper is also in the pool only through another copy
      of that same paper has no independent support. It ranks high.
    * A sole citation, or a claim whose cited papers have no abstract, is more
      exposed: without full text the judgment has nothing to read.
    * "Pool coverage" is BM25 of titles/abstracts of *other* papers. It is an
      ordering factor here, never a skip gate.

    The first pass tries to give each claim one fetchable paper, in claim-risk
    order (when a fetchable citation exists and a slot remains). The second
    pass spends the remaining slots on papers with the highest cumulative
    exposure (then the most claims). Returns deterministic selection metadata;
    the selected papers are the only ones the evidence collector may fetch, so
    deferred papers fall back to their abstracts instead of re-downloading
    serially later.
    """
    rows, records = [], {}
    for c in claims or []:
        ns = list(dict.fromkeys(_cited_for(c, papers)))
        if not ns:
            continue
        titles = [papers[n - 1].get("title") or "" for n in ns]
        support = _pool_support(c["text"], pool, exclude=titles)
        has_abstract = any((papers[n - 1].get("abstract") or "").strip() for n in ns)
        sole = len(ns) == 1
        # A sole citation and an unreadable abstract each add one point; an
        # independent pool match removes up to two more. This keeps coverage a
        # real factor instead of a tie-breaker only.
        risk = ((1.0 if sole else 0.0)
                + (1.0 if not has_abstract else 0.0)
                + 2.0 / (1.0 + support))
        rows.append({"risk": risk, "support": support, "cid": c["id"], "ns": ns})
        for n in ns:
            rec = records.setdefault(n, {"claims": set(), "sole": 0,
                                         "no_abstract": 0, "coverage": 0.0,
                                         "exposure": 0.0})
            rec["claims"].add(c["id"])
            rec["sole"] += 1 if sole else 0
            rec["no_abstract"] += 0 if (papers[n - 1].get("abstract") or "").strip() else 1
            rec["coverage"] += support
            rec["exposure"] += risk

    considered = len(records)
    candidates = [n for n in records if _fetchable(papers[n - 1])]
    candidate_set = set(candidates)
    limit = _fulltext_budget_limit(len(rows), len(candidates))
    if not fulltext.ENABLED:
        limit = 0

    def order(ns):
        return sorted(ns, key=lambda n: (
            -records[n]["exposure"], -len(records[n]["claims"]),
            -records[n]["sole"], -records[n]["no_abstract"],
            records[n]["coverage"], n))

    selected, seen = [], set()

    def take(n):
        if n not in seen and len(selected) < limit:
            seen.add(n)
            selected.append(n)

    # one first source per claim, weakest-evidence claims first
    for row in sorted(rows, key=lambda r: (-r["risk"], r["support"], r["cid"])):
        for n in order(row["ns"]):
            if n in candidate_set and n not in seen:
                take(n)
                break
    # remaining slots: papers that cover the most / least-covered claims
    for n in order(candidates):
        take(n)

    return {
        "mode": "disabled" if not fulltext.ENABLED else str(FULLTEXT_BUDGET),
        "budget": limit,
        "considered": considered,
        "fetchable": len(candidates),
        "no_address": considered - len(candidates),
        "selected": len(selected),
        "deferred": max(0, len(candidates) - len(selected)),
        "selected_indices": selected,
    }


def retrieve_evidence(claim, pool):
    """Search for one claim's extra support; merge new papers into the pool."""
    query = _query_from_claim(claim["text"])
    if not query:
        return 0
    fresh = search.search_many([query], limit=EVIDENCE_PER_CLAIM,
                               sources=EVIDENCE_SOURCES).get(query, [])
    added = 0
    for p in fresh:
        if not p["abstract"]:
            continue
        key = verify.normalize(p["title"])
        if key and key not in pool:
            pool[key] = p
            added += 1
    return added


def retrieve_evidence_many(claims, pool):
    """Search outside support for every claim and merge the new papers into the
    pool. Returns {claim_id: added_count}.

    Measured on a real answer (8 claims, four key-free sources): 17.2s and 43
    papers the pool did not have, 17 of them >=100 citations -- the anchor
    papers a review is measured against. The queries run concurrently, so
    cutting them saves seconds, not minutes: ordering by pool coverage (see
    _pool_support) only decides which ones go when the budget is hit.
    """
    added_by_claim = {c["id"]: 0 for c in claims}
    entries, seen = [], set()
    for c in claims:
        q = _query_from_claim(c["text"])
        if not q or q in seen:      # two claims can collapse to one query
            continue
        seen.add(q)
        entries.append((_pool_support(c["text"], pool), q, c["id"]))
    if not entries:
        return added_by_claim
    entries.sort(key=lambda e: e[0])        # least-covered claim first
    if len(entries) > EVIDENCE_QUERY_BUDGET:
        print(f"      Evidence search: {len(entries)} queries, keeping "
              f"{EVIDENCE_QUERY_BUDGET} (CLAIM_EVIDENCE_QUERY_BUDGET); "
              f"dropping the claims the pool covers best")
        entries = entries[:EVIDENCE_QUERY_BUDGET]
    unique = {q: cid for _, q, cid in entries}
    _check_cancel()
    results = search.search_many(list(unique), limit=EVIDENCE_PER_CLAIM,
                                 sources=EVIDENCE_SOURCES)
    for q, fresh in results.items():
        cid = unique.get(q)
        if cid is None:
            continue
        for p in fresh:
            if not p["abstract"]:
                continue
            key = verify.normalize(p["title"])
            if key and key not in pool:
                pool[key] = p
                added_by_claim[cid] += 1
    return added_by_claim


def fetch_failed(note):
    """True when a full-text attempt was made and produced no text.

    The notes come from fulltext.get_fulltext (see _FETCH_FAILURES). A network
    error, a publisher refusing the PDF or an extraction that ran out of time
    all leave the judge with the abstract alone -- which is a check we could not
    run, not a result. Public because the UI re-derives the same classification
    for snapshots saved before badges existed.
    """
    low = (note or "").lower()
    return any(marker in low for marker in _FETCH_FAILURES)


def _support_badge(supported, supporting, evidence, judged=True):
    """Classify how strong the support for one claim is.

    Returns (key, label, note). The key is stable for the UI to colour-code:
      cross       -- >=2 independent papers back the claim
      fulltext    -- single paper, but we read its full text
      abstract    -- single paper, judged from the abstract only
      unknown     -- judged supported, but no source text was retrieved
      unsupported -- not backed by the evidence
      unfetched   -- judged unsupported, but a cited paper's text never
                     arrived, so the verdict rests on absent evidence
      unjudged    -- the judge returned no usable verdict; this is not a
                     negative result and must not be shown as 不支持
    """
    if not judged:
        return (
            "unjudged", "无法核对（判决未完成）",
            "LLM 未返回可解析的判决——这不是“不支持”，需要重跑或人工核对",
        )
    if not supported:
        failed = [e for e in evidence if fetch_failed(e.get("note"))]
        if failed:
            # seen on a real run: a claim about scPILOT was judged 不支持 while
            # the only evidence record for it said "download failed: HTTP
            # Error 403" -- the paper's text was never read, so the verdict was
            # about a document nobody had opened
            notes = "; ".join(dict.fromkeys(e["note"] for e in failed if e.get("note")))
            return ("unfetched", "无法核对（原文未取到）",
                    f"本次未取到可核对原文（{notes}），判决无从核对——不等于不支持")
        return "unsupported", "不支持", "判决：该声明未被证据支持"
    sup = set(supporting)
    sup_ev = [e for e in evidence if e.get("n") in sup]
    full = [e for e in sup_ev if e.get("source") == "full-text"]
    abstract = [e for e in sup_ev if e.get("source") == "abstract"]
    if len(sup) >= 2:
        note = f"{len(sup)} 篇论文共同支持"
        if sup_ev:
            note += f"，已核对 {len(sup_ev)} 篇（全文 {len(full)} / 摘要 {len(abstract)}）"
        else:
            note += "，未取到可核对原文"
        return "cross", "多篇交叉支持", note
    if full:
        return "fulltext", "全文支持", "支持论文核对到全文原文段落"
    if abstract:
        return "abstract", "仅摘要支持", "支持论文仅核对到摘要，未取到开放全文"
    return "unknown", "支持（未取到原文）", "判决为支持，但未取到支持论文的可核对原文"


def _evidence_meta(paper, n, in_answer):
    """Fields every evidence entry carries, whatever the source of the text."""
    return {
        "n": n,
        "title": paper["title"],
        "in_answer": bool(in_answer),
        "retracted": bool(paper.get("retracted")),
        "retraction_note": paper.get("retraction_note") or "",
    }


def _paper_evidence(claim, paper, n, allow_fetch=True, in_answer=True):
    """Evidence entry for one paper: full-text passages when we can read them,
    otherwise the abstract. Returns (evidence_or_None, prompt_blocks).

    `n` is the number used by the per-sentence numbering (see _evidence_index);
    `in_answer` is False for papers pulled in by the evidence search, which are
    not part of the answer's own bibliography.
    """
    blocks = []
    if allow_fetch:
        candidates, note = fulltext.retrieve_for_claim(claim, paper, candidates=5)
        passages = candidates[:fulltext.TOP_PASSAGES]
        if rerank.ENABLED and len(candidates) > 1:
            reranked = rerank.rerank_passages(
                claim["text"], candidates, top_k=fulltext.TOP_PASSAGES
            )
            if reranked is not None:
                passages = reranked
        if passages:
            blocks.append(f"Full-text passages from [{n}] {paper['title']} ({note}):")
            blocks.extend(f"  > {p['text']}" for p in passages)
            entry = _evidence_meta(paper, n, in_answer)
            entry.update({
                "source": "full-text",
                "note": note,
                "url": paper.get("url") or "",
                "pdf_url": fulltext.pdf_url_for(paper),
                "passages": [
                    {"text": p["text"], "score": p.get("score")} for p in passages
                ],
            })
            return entry, blocks
    abstract = (paper.get("abstract") or "").strip()
    if not abstract:
        return None, blocks
    entry = _evidence_meta(paper, n, in_answer)
    window = abstract[:verify.ABSTRACT_EVIDENCE_CHARS]
    entry.update({
        "source": "abstract",
        "note": note if allow_fetch else "abstract only",
        "url": paper.get("url") or "",
        "pdf_url": fulltext.pdf_url_for(paper),
        "passages": [{"text": window, "score": None}],
    })
    # Put the recorded reason on the evidence *and* in the judge's prompt. Until
    # now a cited paper with no full text contributed nothing to the prompt but
    # its 400-char line in the shared paper list, so "the whole abstract does
    # not back this claim" was decided on a fraction of the abstract.
    blocks.append(f"Abstract of [{n}] {paper['title']} ({entry['note']}):")
    blocks.append(f"  | {window}")
    return entry, blocks


def _evidence_index(answer_papers, pool):
    """Numbering used by the per-sentence stage.

    The first ``len(answer_papers)`` entries are exactly the list that produced
    the answer's [n] markers, so [n] always resolves to the paper the generator
    actually cited. Papers found by the per-claim evidence search are appended
    after them and flagged ``in_answer=False`` in the output.
    """
    index = list(answer_papers)
    seen = {verify.normalize(p["title"]) for p in index}
    for p in (pool or {}).values():
        key = verify.normalize(p["title"])
        if key and key not in seen:
            seen.add(key)
            index.append(p)
    return index


def _cited_for(claim, answer_papers):
    """[n] of one claim, validated against the answer's own numbering.

    A number the generator could not have produced (a hallucinated [99] in a
    20-paper answer) is dropped instead of silently resolving to whichever
    evidence paper happens to sit at that offset.
    """
    return [
        n for n in claim.get("citations", [])
        if isinstance(n, int) and 1 <= n <= len(answer_papers)
    ][:2]


def _cited_for_all(claims, answer_papers):
    return {c["id"]: _cited_for(c, answer_papers) for c in claims}


def _support_title(papers, n, answer_count):
    """Title of supporting paper [n], flagged when it is not in the answer's own
    bibliography (i.e. it came from the per-claim evidence search)."""
    if not (1 <= n <= len(papers)):
        return f"[{n}]?"
    title = papers[n - 1]["title"]
    return title if n <= answer_count else f"{title}（补充检索）"


def _collect_evidence(claims, papers, answer_count, fulltext_indices=None):
    """Gather the cited papers' evidence for every claim concurrently.

    `papers` is the full numbering (see _evidence_index); `answer_count` says
    how many leading entries are the answer's own bibliography.
    `fulltext_indices` is the selected set from plan_fulltext_budget; papers
    outside it contribute their abstract without a download attempt. Returns
    {claim_id: (blocks, evidence_items)} preserving citation order.
    """
    cited = _cited_for_all(claims, papers[:answer_count])
    by_id = {c["id"]: c for c in claims}
    tasks = [(by_id[cid], n) for cid, ns in cited.items() for n in ns]
    out = {c["id"]: ([], []) for c in claims}
    if not tasks:
        return out
    workers = max(1, min(EVIDENCE_WORKERS, len(tasks)))

    def one(claim, n):
        allow = fulltext_indices is None or n in fulltext_indices
        return _paper_evidence(claim, papers[n - 1], n, allow_fetch=allow,
                               in_answer=(n <= answer_count))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # copy_context() per task: a Context cannot be entered by two threads at
        # once, and the workers below have to see this run's stop hook
        futures = {ex.submit(contextvars.copy_context().run, one, c, n): (c["id"], n)
                   for c, n in tasks}
        gathered = {}
        for fut in as_completed(futures):
            cid, n = futures[fut]
            try:
                gathered[(cid, n)] = fut.result()
            except Exception as e:
                print(f"      [warn] evidence for claim {cid} paper [{n}] failed: {e}")
                gathered[(cid, n)] = (None, [])
    for c, n in tasks:
        item, blk = gathered.get((c["id"], n), (None, []))
        blocks, items = out[c["id"]]
        if item:
            items.append(item)
        blocks.extend(blk)
    return out


def _top_up_evidence(claims, verdicts, evidence_by_claim, papers, answer_count):
    """After judging we know which papers actually back each claim. Gather
    evidence for supporting papers that were not cited (fetching full text
    while a small budget lasts, abstract otherwise) so the strength badge
    reflects the supporting paper rather than the cited one."""
    budget = EXTRA_EVIDENCE_BUDGET
    attempted = succeeded = 0
    for c in claims:
        _check_cancel()      # this loop can still download per supporting paper
        v = verdicts.get(c["id"]) or {}
        if not v.get("supported"):
            continue
        ev = evidence_by_claim.setdefault(c["id"], [])
        have = {e.get("n") for e in ev}
        fetched_here = 0
        for n in v.get("supporting", []) or []:
            try:
                n = int(n)
            except (TypeError, ValueError):
                continue
            if n in have or not (1 <= n <= len(papers)):
                continue
            paper = papers[n - 1]
            # spend a download only when an open full text actually exists and
            # this claim has not already spent its share
            allow = (budget > 0 and fetched_here < EXTRA_PER_CLAIM
                     and bool(fulltext.pdf_url_for(paper)))
            if allow:
                budget -= 1
                fetched_here += 1
                attempted += 1
            item, _ = _paper_evidence(c, paper, n, allow_fetch=allow,
                                      in_answer=(n <= answer_count))
            if item:
                if item.get("source") == "full-text":
                    succeeded += 1
                ev.append(item)
                have.add(n)
    if attempted:
        print(f"      Extra full-text lookups: {attempted} attempted, {succeeded} succeeded "
              f"(budget left {budget})")
    return evidence_by_claim


def judge_claims(claims, papers, answer_count=None, fulltext_indices=None):
    """Judge every claim against `papers` (see _evidence_index for the numbering).

    `answer_count` is the size of the answer's own bibliography; supporting
    numbers above it belong to evidence-only papers and are labelled as such.
    `fulltext_indices` limits downloads to the papers selected by
    plan_fulltext_budget; None keeps the old "try every cited paper" behaviour.
    """
    if answer_count is None:
        answer_count = len(papers)
    verdicts = {}
    unjudged = set()
    parse_failures = []
    evidence_by_claim = {}
    passages_used = 0

    # 1) evidence for every claim's cited papers, in parallel
    _check_cancel()
    collected = _collect_evidence(claims, papers, answer_count, fulltext_indices)
    for c in claims:
        blocks, items = collected.get(c["id"], ([], []))
        evidence_by_claim[c["id"]] = items
        passages_used += sum(1 for b in blocks if b.startswith("  > "))

    papers_block = ["Papers:"]
    for i, p in enumerate(papers, 1):
        abstract = (p["abstract"] or "")[:400]
        papers_block.append(f"[{i}] {p['title']} ({p['year']})\nAbstract: {abstract}")

    batches = [claims[i:i + 3] for i in range(0, len(claims), 3)]

    def run_batch(batch):
        blocks = []
        for c in batch:
            blocks.append(f"Claim {c['id']}: {c['text']}")
            blocks.extend(collected.get(c["id"], ([], []))[0])
        blocks.extend(papers_block)
        content = llm.chat(
            [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": "\n\n".join(blocks)},
            ],
            max_tokens=2048,
            temperature=0,
        )
        return batch, content

    # 2) batch judgements, also in parallel
    _check_cancel()
    workers = max(1, min(JUDGE_WORKERS, len(batches)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(contextvars.copy_context().run, run_batch, b): b
                   for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            try:
                _, content = fut.result()
            except Exception as e:
                parse_failures.append(f"judge call failed: {e}")
                unjudged.update(c["id"] for c in batch)
                continue
            batch_verdicts = llm.extract_verdicts(content, id_field="id")
            if not batch_verdicts:
                parse_failures.append(f"claims {[c['id'] for c in batch]}")
                unjudged.update(c["id"] for c in batch)
                continue
            for v in batch_verdicts:
                try:
                    verdicts[int(v["id"])] = v
                except (KeyError, TypeError, ValueError):
                    continue

    if passages_used:
        print(f"      Full-text passages fed to judge: {passages_used}")
    _top_up_evidence(claims, verdicts, evidence_by_claim, papers, answer_count)

    structured = []
    for c in claims:
        v = verdicts.get(c["id"])
        judged = (c["id"] not in unjudged and isinstance(v, dict)
                  and isinstance(v.get("supported"), bool))
        supported = bool(v.get("supported")) if judged else False
        supporting = []
        if judged:
            for n in v.get("supporting", []):
                try:
                    supporting.append(int(n))
                except (TypeError, ValueError):
                    pass
        reason = str(v.get("reason", "")).strip() if isinstance(v, dict) else ""
        evidence = evidence_by_claim.get(c["id"], [])
        badge, badge_label, badge_note = _support_badge(
            supported, supporting, evidence, judged=judged)
        structured.append({
            "id": c["id"],
            "text": c["text"],
            "section": c.get("section"),
            "citations": c.get("citations", []),
            "judged": judged,
            "supported": supported,
            "supporting": supporting,
            "badge": badge,
            "badge_label": badge_label,
            "badge_note": badge_note,
            "reason": reason,
            "evidence": evidence,
            "titles": [_support_title(papers, n, answer_count) for n in supporting],
        })

    lines = []
    for label in parse_failures:
        if label.startswith("judge call failed:"):
            lines.append(("bad", "LLM judge call failed: "
                          + label.split(":", 1)[1].strip()))
        else:
            lines.append(("bad", f"LLM returned no parseable verdicts for {label}"))
    for item in structured:
        c, supported, supporting, reason, titles = (
            item["text"], item["supported"], item["supporting"], item["reason"], item["titles"],
        )
        if supported:
            lines.append(("ok", f"Claim {item['id']}: SUPPORTED ({', '.join(titles) or 'cited papers'})"))
        elif item["badge"] in ("unfetched", "unjudged"):
            # not a failure of the answer: the check did not happen, so the log
            # says so instead of reporting the claim as unsupported
            kind = "UNCHECKED" if item["badge"] == "unfetched" else "UNJUDGED"
            lines.append((
                "warn",
                f"Claim {item['id']}: {kind} -> {c[:110]}"
                f" | {item['badge_note']}",
            ))
        else:
            lines.append((
                "bad",
                f"Claim {item['id']}: UNSUPPORTED -> {c[:110]} | {reason}",
            ))
    return lines, structured, parse_failures, passages_used


def _cited_papers(claims, papers):
    """Papers the claims point at, in answer numbering (1-based)."""
    idx = set()
    for c in claims:
        for n in c.get("citations", []):
            if isinstance(n, int) and 1 <= n <= len(papers):
                idx.add(n)
    return [papers[i - 1] for i in sorted(idx)]


def _offer_alternative_urls(papers):
    """Ask OpenAlex for an open-access copy of every cited paper that needs one.

    Two groups qualify. A paper with no PDF URL at all never gets a download
    attempt: on the ten saved surveys that was 84 of 243 cited papers (35%).
    A paper whose URL already failed (the marker on disk says so) is worth a
    second address -- "HTTP Error 403: Forbidden" is the most common reason in
    the skip markers (50 of 102), and publishers point bots at a landing page
    while the copy in a repository stays open. One batched request per 50 DOIs
    costs a second or so, and papers that already have a working address are
    not asked about. Never fatal: no answer just means the stage runs as before.
    """
    needed = [p for p in papers if not fulltext.pdf_url_for(p) or fulltext.is_known_bad(p)]
    if not needed:
        return
    try:
        mapping = search.openalex_oa_pdf_urls(needed)
    except Exception as e:
        print(f"      [warn] OpenAlex open-access lookup failed ({e}); continuing")
        return
    fulltext.set_oa_urls(mapping)
    if mapping:
        print(f"      Open-access fallback: {len(mapping)}/{len(needed)} of the papers "
              f"with no working URL got a second address from OpenAlex")


def run(answer, papers, pool=None, deadline_s=None, should_cancel=None):
    """Full per-sentence check.

    `answer` is the generated text; `papers` MUST be the same ordered list the
    answer's [n] markers were built from (graph state["papers"]) -- that is what
    makes every citation resolve to the paper the generator actually cited.
    `pool` is the wider retrieval pool searched for extra supporting evidence;
    papers found there are numbered after the answer's own papers.

    Returns {"lines": report lines for the log, "verdicts": structured claim
    verdicts, "passages_used": int, "parse_failures": [...],
    "fulltext_stats": {...}}.

    `deadline_s` caps the whole full-text stage; lookups past it fall back to
    abstracts instead of blocking the answer.

    `should_cancel` is a callable the caller may pass in: it is invoked at the
    points where stopping is cheap (before each batch of evidence queries,
    before the full-text prefetch, between judging stages) and is expected to
    raise when the user asked to stop.
    """
    if isinstance(papers, dict):
        raise TypeError(
            "sentences.run(answer, papers, pool=None): papers must be the "
            "answer's own numbered list (state['papers']), not the pool dict"
        )
    fulltext.reset_stats()
    fulltext.set_deadline(deadline_s if deadline_s is not None else fulltext.DEADLINE_S)
    # a context variable, not a module global: this stage is where a stop is
    # most expensive to notice, and with two runs overlapping in one process a
    # single slot would leave the other run checking somebody else's flag
    token = _cancel_hook.set(should_cancel)
    try:
        return _run(answer, list(papers or []), pool if pool is not None else {})
    finally:
        _cancel_hook.reset(token)
        try:
            fulltext.clear_deadline()
        except Exception:
            pass


# set by run(should_cancel=...) so a stop is noticed early. It reaches the
# parallel workers because every pool below is handed a copy of this context.
_cancel_hook = contextvars.ContextVar("sentences_cancel_hook", default=None)


def _check_cancel():
    hook = _cancel_hook.get()
    if hook:
        hook()


def _run(answer, papers, pool):
    _check_cancel()      # the caller may have stopped us before the first LLM call
    print("      Splitting answer into claims...")
    claims = split_report_claims(answer)
    if not claims:
        empty = ["--- Per-sentence evidence check ---", "      no claims extracted"]
        return {"lines": empty, "verdicts": [], "passages_used": 0,
                "parse_failures": [], "fulltext_stats": fulltext.stats_summary()}
    print(f"      {len(claims)} claims; retrieving evidence per claim...")
    _check_cancel()
    added_by_claim = retrieve_evidence_many(claims, pool)
    for cid, added in added_by_claim.items():
        if added:
            print(f"        claim {cid}: +{added} new papers")
    # answer papers first, evidence-only papers appended: an [n] in a claim
    # keeps the meaning it had in the answer.
    index = _evidence_index(papers, pool)
    # download the cited papers' full texts up front, in parallel, so the
    # judging loop below hits the cache instead of paying N serial downloads
    cited = _cited_papers(claims, papers)
    plan = plan_fulltext_budget(claims, papers, pool)
    if cited:
        # downloads are the slowest single step of this stage: check first so a
        # stop does not sit through the whole prefetch
        _check_cancel()
        _offer_alternative_urls(cited)
        # OpenAlex may have supplied a second address for a paper that had
        # none, so the plan is made after that lookup.
        plan = plan_fulltext_budget(claims, papers, pool)
        selected = [papers[n - 1] for n in plan["selected_indices"]]
        if plan["considered"]:
            print(f"      Full-text budget: {plan['selected']}/{plan['fetchable']} "
                  f"fetchable cited papers selected (mode {plan['mode']}, "
                  f"deferred {plan['deferred']} to abstract-only)")
        if selected:
            t0 = time.monotonic()
            scheduled = fulltext.prefetch_fulltexts(selected)
            if scheduled:
                print(f"      Prefetched {scheduled} full texts in {time.monotonic() - t0:.1f}s")
    print("      Judging claim support...")
    _check_cancel()
    lines, structured, parse_failures, passages_used = judge_claims(
        claims, index, answer_count=len(papers),
        fulltext_indices=plan["selected_indices"],
    )
    stats = fulltext.stats_summary()
    stats["budget"] = {k: v for k, v in plan.items() if k != "selected_indices"}
    print(f"      Full text: {stats['cache_hits'] + stats['memo_hits']} cache hits, "
          f"{stats['downloads']} downloads, {stats['download_failed']} failed, "
          f"{stats['download_seconds']:.1f}s downloading "
          f"(hit rate {stats['cache_hit_rate']:.0%})")
    report_lines = ["--- Per-sentence evidence check ---"] + [
        f"{'OK ' if status == 'ok' else 'BAD'} {msg}" for status, msg in lines
    ]
    return {
        "lines": report_lines,
        "verdicts": structured,
        "passages_used": passages_used,
        "parse_failures": parse_failures,
        "fulltext_stats": stats,
        # the evidence search may have merged new papers into `pool`; the caller
        # decides what to do with them (the graph node writes it back to state)
        "pool": pool,
    }
