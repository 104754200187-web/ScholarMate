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

MAX_CLAIMS = 8
EXTRA_EVIDENCE_BUDGET = 4  # global cap on full-text fetches for supporting-but-uncited papers
EXTRA_PER_CLAIM = 1        # at most one such fetch per claim (enough to prove full-text support)
# The per-sentence stage is dominated by LLM calls (passage rerank per cited
# paper, then the batch judgement). They are independent, so run them in
# parallel; DeepSeek tolerates this comfortably at 4-5 in flight.
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

SPLIT_PROMPT = """Split the following literature answer into its key factual \
claims. Each claim should be a single statement asserting something about the \
literature (usually a sentence or clause, keeping any citation [n]). Reply \
ONLY with JSON:
{"claims": [{"id": 1, "text": "...", "citations": [n, ...]}]}
Rules: max 8 claims; merge trivial filler sentences; keep exact wording."""

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


def split_claims(answer):
    content = llm.chat(
        [
            {"role": "system", "content": SPLIT_PROMPT},
            {"role": "user", "content": answer},
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
    claims = []
    for c in result.get("claims", []):
        text = str(c.get("text", "")).strip()
        if not text:
            continue
        try:
            cid = int(c.get("id", len(claims) + 1))
        except (TypeError, ValueError):
            cid = len(claims) + 1
        citations = []
        for n in c.get("citations", []):
            try:
                citations.append(int(n))
            except (TypeError, ValueError):
                pass
        claims.append({"id": cid, "text": text, "citations": citations})
    return claims[:MAX_CLAIMS]


def _query_from_claim(text):
    query = re.sub(r"\[\d+\]", "", text).strip()
    return query[:120]


def _pool_support(claim_text, pool):
    """Best BM25 score of any pooled paper against one claim.

    Used to *order* the evidence queue: when the budget forces a cut, the
    claims the pool already covers best are the ones to drop. It is deliberately
    not a skip gate -- measured, dropping covered claims saved a few seconds and
    lost ~40 papers, because a claim's own paper is not the only relevant one.
    """
    papers = [p for p in (pool or {}).values() if p.get("title")]
    if not papers:
        return 0.0
    docs = [fulltext.tokens((p.get("title") or "") + " " + (p.get("abstract") or ""))
            for p in papers]
    scores = fulltext.bm25_scores(claim_text, docs)
    return max(scores) if scores else 0.0


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


def _support_badge(supported, supporting, evidence):
    """Classify how strong the support for one claim is.

    Returns (key, label, note). The key is stable for the UI to colour-code:
      cross       -- >=2 independent papers back the claim
      fulltext    -- single paper, but we read its full text
      abstract    -- single paper, judged from the abstract only
      unknown     -- judged supported, but no source text was retrieved
      unsupported -- not backed by the evidence
    """
    if not supported:
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
    entry.update({
        "source": "abstract",
        "note": note if allow_fetch else "abstract only",
        "url": paper.get("url") or "",
        "pdf_url": fulltext.pdf_url_for(paper),
        "passages": [{"text": abstract[:600], "score": None}],
    })
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


def _collect_evidence(claims, papers, answer_count):
    """Gather the cited papers' evidence for every claim concurrently.

    `papers` is the full numbering (see _evidence_index); `answer_count` says
    how many leading entries are the answer's own bibliography. Returns
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
        return _paper_evidence(claim, papers[n - 1], n, in_answer=(n <= answer_count))

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


def judge_claims(claims, papers, answer_count=None):
    """Judge every claim against `papers` (see _evidence_index for the numbering).

    `answer_count` is the size of the answer's own bibliography; supporting
    numbers above it belong to evidence-only papers and are labelled as such.
    """
    if answer_count is None:
        answer_count = len(papers)
    verdicts = {}
    parse_failures = []
    evidence_by_claim = {}
    passages_used = 0

    # 1) evidence for every claim's cited papers, in parallel
    _check_cancel()
    collected = _collect_evidence(claims, papers, answer_count)
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
        futures = [ex.submit(contextvars.copy_context().run, run_batch, b)
                   for b in batches]
        for fut in as_completed(futures):
            try:
                batch, content = fut.result()
            except Exception as e:
                parse_failures.append(f"judge call failed: {e}")
                continue
            batch_verdicts = llm.extract_verdicts(content, id_field="id")
            if not batch_verdicts:
                parse_failures.append(f"claims {[c['id'] for c in batch]}")
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
        v = verdicts.get(c["id"], {})
        supported = bool(v.get("supported"))
        supporting = []
        for n in v.get("supporting", []):
            try:
                supporting.append(int(n))
            except (TypeError, ValueError):
                pass
        reason = str(v.get("reason", "")).strip()
        evidence = evidence_by_claim.get(c["id"], [])
        badge, badge_label, badge_note = _support_badge(supported, supporting, evidence)
        structured.append({
            "id": c["id"],
            "text": c["text"],
            "citations": c.get("citations", []),
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
        lines.append(("bad", f"LLM returned no parseable verdicts for {label}"))
    for item in structured:
        c, supported, supporting, reason, titles = (
            item["text"], item["supported"], item["supporting"], item["reason"], item["titles"],
        )
        if supported:
            lines.append(("ok", f"Claim {item['id']}: SUPPORTED ({', '.join(titles) or 'cited papers'})"))
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
    claims = split_claims(answer)
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
    if cited:
        # downloads are the slowest single step of this stage: check first so a
        # stop does not sit through the whole prefetch
        _check_cancel()
        t0 = time.monotonic()
        scheduled = fulltext.prefetch_fulltexts(cited)
        if scheduled:
            print(f"      Prefetched {scheduled} full texts in {time.monotonic() - t0:.1f}s")
    print("      Judging claim support...")
    _check_cancel()
    lines, structured, parse_failures, passages_used = judge_claims(
        claims, index, answer_count=len(papers)
    )
    stats = fulltext.stats_summary()
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
