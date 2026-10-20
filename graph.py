"""LangGraph orchestration for the literature agent.

Graph: plan_queries -> retrieve -> rerank -> screen_retractions -> score_evidence
       -> generate -> critique (loop back to retrieve while insufficient,
       max MAX_ROUNDS) -> verify_rules -> verify_llm -> verify_sentences -> END
       score_evidence also routes to format_papers (search mode) or END (empty
       pool) instead of generate.

Reuses the pure-stdlib primitives in search / llm / verify / sentences /
rerank / fulltext; only the orchestration layer is rewritten with LangGraph.
"""

import operator
import os
import threading
import time
import uuid
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

import chat
import explore
import journal
import llm
import rerank
import retraction
import scoring
import search
import sentences
import verify

MAX_ROUNDS = 3  # initial round + up to 2 follow-up rounds
# LangGraph refuses to start a run without one, and the default (25) sits just
# three steps above our longest path (22 node visits at MAX_ROUNDS=3).
RECURSION_LIMIT = int(os.environ.get("LANGGRAPH_RECURSION_LIMIT", str(6 * MAX_ROUNDS + 12)))
# Retraction screening is worth its ~13s when writing an answer (you must not
# build a review on a retracted paper) but it is 29% of a search-only run: off
# there by default, on request (UI checkbox) or via env for every run.
RETRACTION_IN_SEARCH = os.environ.get("RETRACTION_IN_SEARCH", "0") != "0"
PAPERS_PER_QUERY = 8
# LangGraph checkpointer: every super-step is snapshotted per thread, so a run
# that dies (a DeepSeek outage, a killed server) can be resumed from the last
# completed node instead of paying for the whole retrieval again. In-memory
# only -- it survives a failed *run*, not a restarted process.
CHECKPOINT_ENABLED = os.environ.get("LANGGRAPH_CHECKPOINT", "1") != "0"
# each checkpoint holds a whole paper pool, so bound how many threads stay hot
MAX_THREADS = int(os.environ.get("LANGGRAPH_MAX_THREADS", "8"))
MAX_TOTAL_PAPERS = 20
# a review spans several directions at once, so its working set is
# wider than a single question's: still bounded, but not 20.
SURVEY_MAX_PAPERS = int(os.environ.get("SURVEY_MAX_PAPERS", "30"))
# Citation-ranked companion pass (see search.search_classics): run on the first
# round only, over the first CLASSIC_QUERIES queries, so the extra coverage is
# bounded instead of doubling every round.
CLASSIC_QUERIES = 2
# A follow-up turn that lacks material fetches it itself (see
# node_chat_retrieve) instead of telling the user to start a whole new run.
# Two passes is the point where an answer stops improving in practice.
CHAT_MAX_ROUNDS = int(os.environ.get("CHAT_MAX_ROUNDS", "2"))
# papers a follow-up may add to the working set in one turn, and the ceiling
# the conversation's paper list may reach (numbers are append-only, so the
# list only ever grows)
CHAT_MAX_NEW_PAPERS = int(os.environ.get("CHAT_MAX_NEW_PAPERS", "8"))
CHAT_MAX_TOTAL_PAPERS = int(os.environ.get("CHAT_MAX_TOTAL_PAPERS", "48"))

# Sort options for search-only mode ("relevance" keeps the LLM rerank).
# Each entry maps a stable key to (paper field, reverse) used by sort_papers.
SEARCH_SORTS = {
    "relevance": None,
    "year_desc": ("year", True),    # 发表时间：近 -> 远
    "year_asc": ("year", False),    # 发表时间：远 -> 近
    "cited_desc": ("citations", True),
    "cited_asc": ("citations", False),
    "title_asc": ("title", False),
}
SEARCH_SORT_LABELS = {
    "relevance": "相关性",
    "year_desc": "发表时间 近→远",
    "year_asc": "发表时间 远→近",
    "cited_desc": "被引量 高→低",
    "cited_asc": "被引量 低→高",
    "title_asc": "标题 A→Z",
}

SYSTEM_PROMPT = """You are a scientific literature assistant. Answer the user's \
question based ONLY on the provided papers. Rules:
1. Cite sources inline as [n], matching the paper numbers given.
2. Never invent papers or citations. If the provided papers are insufficient, \
say so explicitly.
3. End with a "References" section listing ONLY the papers you actually cited \
in the body, in the form [n] Title (Year). Do not list uncited papers.
4. Every [n] cited in the body MUST appear in the References section.
5. Answer in the same language as the user's question.
6. A paper flagged RETRACTED must never be used as evidence. Cite it only to \
say that it has been retracted, and prefer non-retracted papers instead."""

SURVEY_PROMPT = """你是 ScholarMate 的综述写作者。用户在做一次领域调研，需要你把下面这批论文写成一份**结构化的文献综述报告**，而不是回答一个问题。

写作要求：
1. 用中文写 Markdown，结构固定为：`## 摘要`（150-250 字，说明覆盖了哪几个方向、主要结论是什么）→ 每个方向一个 `## ` 小节 → `## 争议与空白` → `## 参考文献`。
2. 小节顺序与【报告目录】一致；小节标题写成「中文方向名（English term）」。
3. 所有事实性陈述必须带 [n] 行内引用，编号严格对应下面的论文表。不得引入表外的任何文献，也不得编造样本量、实验条件或结论强度。
4. 每个方向小节写 2-4 段：这个方向在解决什么问题、主流方法、代表性结果、目前的局限；结论不一致时如实写出分歧，不要强行统一。
5. 本次检索没有覆盖到的方面，直接写「本次检索未覆盖」，不要用空话填满篇幅。
6. `## 参考文献` 只列正文真正引用过的论文，写成 `[n] 标题 (年份)`。
7. 不要写任何与报告无关的寒暄或自我说明。"""

CRITIC_PROMPT = """You are a research assistant checking whether the retrieved \
evidence is sufficient to answer the question thoroughly. Reply ONLY with a JSON \
object, no other text:
{"sufficient": true/false, "follow_up_queries": ["...", "..."]}

- Set sufficient=false if important aspects are missing, claims lack support, \
or more recent/authoritative evidence would materially improve the answer.
- When insufficient, provide 1-3 concise search queries (English) targeting the \
missing evidence. Otherwise return an empty list."""

QUERY_PLANNER_PROMPT = """You are a search query planner for a scientific \
literature agent. Given the research question, output 2-3 concise English \
search queries that would retrieve relevant academic papers from a \
bibliographic database such as OpenAlex (which handles English much better \
than other languages). Reply ONLY with JSON: {"queries": ["...", "..."]}."""


# Progress callbacks. They are stored per run (keyed by the run the calling
# thread announced with begin_run()) instead of in two module globals: with two
# runs overlapping in one process, the second run's install would silently
# retarget the first one's progress lines, and the pairs would be restored in
# the wrong order on the way out. The plain globals stay as the fallback for a
# run that was never registered -- the CLI.
_stage_hook = None    # set by run_full(on_stage=...) so the web UI can show progress
_timing_hook = None   # set by run_full(on_timing=...) so slow stages are visible
_hooks = {}           # run key -> (on_stage, on_timing)
_hooks_lock = threading.Lock()

_checkpointer = InMemorySaver() if CHECKPOINT_ENABLED else None


# ------------------------------------------------------- stopping a run
# Python threads cannot be killed, so a stop request is cooperative: the web
# layer sets an event and the pipeline raises at its next checkpoint (every
# node boundary, every round of retrieval, every claim of the sentence check).
# The event is keyed by the run's thread id rather than being one module-level
# flag, because two people using the web UI at once must not be able to stop
# each other's run. A worker announces the run it is executing with begin_run(),
# so the deep call sites below (_emit, the node wrapper, the per-sentence hook)
# find their own event without a job id threaded through every function.
class JobCancelled(Exception):
    """Raised inside a run when the web layer asks it to stop."""


_cancel_events = {}
_cancel_lock = threading.Lock()
_current_run = threading.local()


def begin_run(key):
    """Register `key` as the run the calling thread is executing.

    Returns a zero-argument callable that ends the registration, so a worker
    can hand it straight to a `finally`.

    An event that is already registered is *kept*, not reset: someone may have
    asked to stop this run before the worker got to it (a queued job whose stop
    button was pressed). A key cannot carry a stale event from an earlier run,
    because end_run() removes the entry.
    """
    key = str(key)
    with _cancel_lock:
        event = _cancel_events.get(key)
        if event is None:
            event = threading.Event()
            _cancel_events[key] = event
    _current_run.key = key

    def done():
        end_run(key)

    return done


def end_run(key=None):
    """Forget a finished run's event so the registry does not grow forever."""
    current = getattr(_current_run, "key", None)
    target = current if key is None else str(key)
    if key is None or current == target:
        _current_run.key = None
    if target is None:
        return
    with _cancel_lock:
        _cancel_events.pop(target, None)
    with _hooks_lock:
        _hooks.pop(target, None)


def request_cancel(key):
    """Ask the run `key` to stop at its next checkpoint."""
    with _cancel_lock:
        event = _cancel_events.get(str(key))
        if event is None:
            event = threading.Event()
            _cancel_events[str(key)] = event
    event.set()


def cancel_requested(key=None):
    event = _event_for(key)
    return bool(event and event.is_set())


def _event_for(key=None):
    if key is None:
        key = getattr(_current_run, "key", None)
    if key is None:
        return None
    with _cancel_lock:
        return _cancel_events.get(str(key))


def _check_cancel():
    event = _event_for()
    if event is not None and event.is_set():
        raise JobCancelled("已按用户请求停止")


def install_hooks(on_stage, on_timing):
    """Point the run this thread is executing at its own progress callbacks."""
    global _stage_hook, _timing_hook
    key = getattr(_current_run, "key", None)
    if key is None:
        _stage_hook, _timing_hook = on_stage, on_timing
        return
    with _hooks_lock:
        _hooks[str(key)] = (on_stage, on_timing)


def clear_hooks():
    """Drop this run's callbacks (end_run() does it too, as a backstop)."""
    global _stage_hook, _timing_hook
    key = getattr(_current_run, "key", None)
    if key is None:
        _stage_hook, _timing_hook = None, None
        return
    with _hooks_lock:
        _hooks.pop(str(key), None)


def _hooks_for():
    """(on_stage, on_timing) of the run this thread is executing."""
    key = getattr(_current_run, "key", None)
    if key is None:
        return _stage_hook, _timing_hook
    with _hooks_lock:
        pair = _hooks.get(str(key))
    return pair if pair else (None, None)


def _config(thread_id=None):
    """LangGraph run config: recursion headroom plus the checkpoint thread."""
    cfg = {"recursion_limit": RECURSION_LIMIT}
    if thread_id:
        cfg["configurable"] = {"thread_id": thread_id}
    return cfg


def checkpoint_info(thread_id):
    """What is stored for one run: {exists, next, checkpoints} (for the UI)."""
    if not (CHECKPOINT_ENABLED and thread_id):
        return {"exists": False, "next": [], "checkpoints": 0}
    try:
        app = get_app()
        state = app.get_state(_config(thread_id))
    except Exception:
        return {"exists": False, "next": [], "checkpoints": 0}
    nxt = [str(n) for n in (getattr(state, "next", ()) or ())]
    values = getattr(state, "values", None) or {}
    try:
        saved = sum(1 for _ in _checkpointer.list(_config(thread_id)))
    except Exception:
        saved = 0
    return {"exists": bool(nxt or values), "next": nxt, "checkpoints": saved}


def has_checkpoint(thread_id):
    """True when a failed run can be picked up where it stopped."""
    info = checkpoint_info(thread_id)
    return bool(info["exists"] and info["next"])


def snapshot_state(thread_id):
    """Best-effort read of the checkpointed state (for a stopped run's summary)."""
    if not (CHECKPOINT_ENABLED and thread_id):
        return {}
    try:
        state = get_app().get_state(_config(thread_id))
    except Exception:
        return {}
    values = getattr(state, "values", None)
    return values if isinstance(values, dict) else {}


def drop_checkpoint(thread_id):
    """Forget one run's snapshots (after a successful one, say)."""
    if not (CHECKPOINT_ENABLED and thread_id):
        return False
    try:
        return bool(_checkpointer.delete_thread(thread_id))
    except Exception:
        return False


def prune_checkpoints(keep=None):
    """Keep only the `keep` most recently written threads (memory is finite)."""
    if not CHECKPOINT_ENABLED:
        return 0
    keep = MAX_THREADS if keep is None else keep
    storage = getattr(_checkpointer, "storage", None)
    if not isinstance(storage, dict) or len(storage) <= keep:
        return 0
    victims = list(storage.keys())[:-keep]
    dropped = 0
    for tid in victims:
        if drop_checkpoint(tid):
            dropped += 1
    return dropped


def _emit(text):
    # a stage boundary is the cheapest place to notice a stop request
    _check_cancel()
    hook = _hooks_for()[0]
    if hook:
        try:
            hook(text)
        except Exception:
            pass


# Human-facing description of every node, kept next to the node definitions so
# the README and the web panel can be generated from the compiled graph instead
# of being maintained by hand.
NODE_INFO = {
    "explore_scope": ("🔭 领域拆解", "把领域（或选中的方向）展开成 3-6 个可分别调研的方向，每个方向取 3 篇高被引代表作",
                      "explore.py / llm.py"),
    "survey_plan": ("🔭 综述检索规划", "把选中方向的检索词轮转合并去重，交给常规检索链", "explore.py"),
    "plan_queries": ("① 查询规划", "DeepSeek 把问题拆成 2-3 条英文检索词（失败退回原问题）", "graph.py / llm.py"),
    "retrieve": ("② 检索", "6 个数据源 × 全部检索词并行检索，跨源去重合并；首轮追加一路按被引量排序的经典锚点检索",
                 "search.py"),
    "rerank": ("③ 重排", "ask：DeepSeek 相关性打分取 top 20；search：按所选字段排序且不截断", "rerank.py"),
    "score_evidence": ("④b 证据评分", "10 个维度逐篇打分（9 维由 DeepSeek 判、影响力本地算），按用户权重合成总分与等级", "scoring.py"),
    "screen_retractions": ("④ 撤稿筛查", "OpenAlex 批量 + Crossref DOI/标题反查，标注已撤稿论文", "retraction.py"),
    "generate": ("⑤ 生成回答", "基于 top-N 论文写带 [n] 行内引用与 References 的综述", "graph.py / llm.py"),
    "format_papers": ("✅ 输出论文列表", "search 模式终点：直接输出论文列表，不做校验（与 generate 互斥）", "graph.py"),
    "critique": ("⑥ 证据充分性评估", "DeepSeek 判断证据是否充分，不足则给出 follow-up 检索词", "graph.py / llm.py"),
    "verify_rules": ("⑦ 三层校验 1/3", "规则层：[n] 是否映射到论文且标题匹配、是否引用了已撤稿论文", "verify.py"),
    "verify_llm": ("⑧ 三层校验 2/3", "归因层：每条引用是否被该论文摘要支持（DeepSeek）", "verify.py"),
    "verify_sentences": ("⑨ 三层校验 3/3", "逐句层：拆声明 → 逐声明取证 → 全文段落 → 判决/强度徽标", "sentences.py / fulltext.py"),
    "chat_answer": ("💬 追问回答", "不重新检索：只用本次工作论文集 + 上一轮回答作答，引用编号与之一致；材料不够时明确要求补检索",
                    "chat.py / llm.py"),
    "chat_retrieve": ("🔎 追问补检索", "追问发现工作集里没有材料时，就地为这一轮补检论文并**追加**编号（已有编号不动）",
                      "graph.py / search.py"),
    "chat_verify": ("💬 追问校验", "对追问回答做规则层校验（[n] 是否越界/错配、是否引用撤稿论文）", "verify.py"),
}


# execution order, so the table in the docs/UI reads like the pipeline
NODE_ORDER = (
    "explore_scope", "survey_plan",
    "plan_queries", "retrieve", "rerank", "screen_retractions", "score_evidence",
    "generate", "format_papers", "critique",
    "verify_rules", "verify_llm", "verify_sentences",
    "chat_answer", "chat_retrieve", "chat_verify",
)


def describe_graph():
    """Topology of the compiled graph, for the web panel and the docs.

    Nodes and edges are read back from what actually executes; only the labels
    are curated. If a node is added to build_graph() it shows up here (and in
    the UI) without anyone remembering to update a diagram.
    """
    g = get_app().get_graph()
    nodes = []
    for nid in sorted(g.nodes, key=lambda n: (NODE_ORDER.index(n), n)
                      if n in NODE_ORDER else (len(NODE_ORDER), n)):
        if nid.startswith("__"):
            continue
        label, note, impl = NODE_INFO.get(nid, (nid, "", ""))
        nodes.append({"id": nid, "label": label, "note": note, "impl": impl})
    known = {n["id"] for n in nodes}
    edges = [
        {"source": e.source, "target": e.target, "conditional": bool(e.conditional)}
        for e in g.edges
        if e.source in known or e.target in known
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "max_rounds": MAX_ROUNDS,
        "papers_per_query": PAPERS_PER_QUERY,
        "max_total_papers": MAX_TOTAL_PAPERS,
        "sources": search.source_status(),
        "scoring": {"dimensions": scoring.dimensions_payload(),
                    "presets": scoring.presets_payload(),
                    "max_papers": scoring.MAX_PAPERS,
                    "batch_size": scoring.BATCH_SIZE},
        "classic": {"per_query": search.CLASSIC_PER_QUERY,
                    "min_citations": search.CLASSIC_MIN_CITATIONS,
                    "queries": CLASSIC_QUERIES},
        "loop": "critique → retrieve",
    }


def _merge_timings(left, right):
    """Add up per-node seconds: one node can run in several attempts."""
    out = dict(left or {})
    for name, seconds in (right or {}).items():
        out[name] = round(out.get(name, 0.0) + seconds, 2)
    return out


class ResearchState(TypedDict):
    question: str
    mode: str                     # "ask" (full pipeline) | "search" (papers only)
    sort: str                     # search-mode ordering; see SEARCH_SORTS
    filters: dict                 # {"min_year", "max_year", "review_only"}
    queries: list[str]
    pool: dict                    # normalized title -> paper
    papers: list[dict]            # reranked working set (top MAX_TOTAL_PAPERS)
    answer: str
    retraction: dict              # retraction screening summary
    screen_retractions: bool      # run retraction screening in this mode?
    score_evidence: bool          # run the multi-dimensional evidence scoring?
    weights: dict                 # per-dimension weights for the evidence score
    target_species: str           # target organism for 物种匹配 ("" = ignore)
    evidence: dict                # evidence scoring summary
    rounds: int                   # retrieval rounds executed
    verdict: dict                 # critique result
    retrieval_log: list           # per-round per-query source stats
    scores: list                  # rerank scores (title -> score) or []
    rule_audit: dict              # structured rule verification
    llm_audit: dict               # structured LLM attribution audit
    claim_audit: dict             # structured per-sentence audit
    deep_verify: bool             # run the per-sentence full-text check?
    reports: Annotated[list[list[str]], operator.add]  # verification blocks
    # --- field exploration (see explore.py) --------------------------------
    explore_action: str           # "scope" | "deepen" | "survey" | "" (ordinary run)
    explore_path: list            # breadcrumb of chosen directions, outermost first
    explore_seen: list            # direction names already shown in this session
    explore_parent: dict          # the direction being deepened (deepen only)
    explore_directions: list      # directions proposed by this turn / chosen for the report
    explore_preview: bool         # attach landmark-paper previews to each direction?
    explore_meta: dict            # degraded flag / counts / duration
    directions: list              # survey runs: the chosen directions (drives the report)
    # --- interactive follow-up turn (see chat.py) --------------------------
    chat_message: str             # the user's new question ("" = ordinary run)
    chat_history: list            # previous turns: [{role, content}, ...]
    chat_answer: str              # this turn's reply
    chat_audit: dict              # rule-layer audit of that reply
    chat_meta: dict               # needs_new_search / search_query / timings
    chat_allow_retrieve: bool     # may this turn fetch papers it is missing?
    chat_rounds: int              # retrieval passes already run this turn
    chat_tried: list              # queries this turn already retrieved for
    chat_new_papers: list         # papers this turn added (append-only numbering)
    chat_log: list                # human-readable trace of the passes
    # per-node wall clock, accumulated by the node wrapper. It lives in the
    # state (instead of being stitched on afterwards) so a resumed run reports
    # the stages the earlier attempt already paid for as well.
    timings: Annotated[dict, _merge_timings]


# ---------------------------------------------------------------- helpers

def build_context(papers):
    blocks = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"][:3])
        abstract = p["abstract"][:800] if p["abstract"] else "(no abstract)"
        blocks.append(
            f"[{i}] {p['title']} ({p['year']})\n"
            f"Authors: {authors}\n"
            f"URL: {p['url']}\n"
            f"Abstract: {abstract}"
            + ("\nStatus: *** RETRACTED PAPER *** do not rely on it as evidence"
               if p.get("retracted") else "")
        )
    return "\n\n".join(blocks)


def format_papers(papers, sort_label="相关性"):
    """Render the retrieved papers as markdown (search-only mode)."""
    lines = [f"## 检索到 {len(papers)} 篇论文（排序：{sort_label}）", ""]
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"][:4]) or "未知"
        lines.append(f"### {i}. {p['title']}")
        lines.append(
            f"**年份**：{p['year'] or '-'} ｜ **被引**：{p.get('citations') or 0}"
            f" ｜ **来源**：{p.get('source', '')}"
            + (f" ｜ **刊物/会议**：{p.get('venue', '')}" if p.get("venue") else "")
        )
        lines.append(f"**作者**：{authors}")
        abstract = p.get("abstract") or ""
        if abstract:
            suffix = "..." if len(abstract) > 400 else ""
            lines.append(abstract[:400] + suffix)
        if p.get("url"):
            lines.append(f"链接：{p['url']}")
        if p.get("pdf_url"):
            lines.append(f"PDF：{p['pdf_url']}")
        if p.get("doi"):
            lines.append(f"DOI：{p['doi']}")
        lines.append("")
    return "\n".join(lines)


def sort_papers(papers, sort):
    """Sort a paper list for search-only mode. Papers missing the sort field
    (e.g. no year on Google Scholar) sink to the bottom regardless of order."""
    spec = SEARCH_SORTS.get(sort)
    if not spec:
        return papers
    field, reverse = spec

    def key(p):
        value = p.get(field)
        # the missing-value flag has to flip with the direction, otherwise
        # reverse=True would sort None to the TOP (it did, for 未评分 papers)
        if reverse:
            return (0, "") if value is None else (1, value)
        return (1, "") if value is None else (0, value)

    return sorted(papers, key=key, reverse=reverse)


def add_papers_many(pool, queries, limit=PAPERS_PER_QUERY, filters=None, classic=False):
    """Retrieve for several queries, merge into the pool (dedup by title) and
    return (added_count, per-query provenance stats for the audit log).

    `classic=True` runs the citation-ranked companion pass instead of the
    relevance pass (same queries, hits ordered by citation count and cut to the
    field's landmark papers); the papers it contributes are tagged so the UI can
    show where they came from.
    """
    if classic:
        details = search.search_classics(queries, limit=limit, filters=filters)
    else:
        details = search.search_many_detailed(queries, limit=limit, filters=filters)
    total = 0
    queries_log = []
    for d in details:
        added = 0
        for p in d["papers"]:
            if not p["abstract"]:
                continue
            key = verify.normalize(p["title"])
            if key and key not in pool:
                if classic:
                    p = dict(p, via_classic=True)
                pool[key] = p
                added += 1
        total += added
        queries_log.append({
            "query": d["query"],
            "sources": d["sources"],
            "hits": len(d["papers"]),
            "classic": bool(classic),
            "error": d.get("error"),
        })
    return total, queries_log


def plan_queries(question):
    """Turn the user question into English retrieval queries."""
    try:
        result = llm.chat_json(
            [
                {"role": "system", "content": QUERY_PLANNER_PROMPT},
                {"role": "user", "content": f"Question: {question}"},
            ],
            max_tokens=256,
            temperature=0,
        )
        queries = [q for q in result.get("queries", []) if q.strip()]
        if queries:
            print(f"      Planned queries: {queries}")
            return queries
    except Exception as e:
        print(f"      Query planning failed ({e}); using original question")
    return [question]


def criticize(question, papers, answer):
    messages = [
        {"role": "system", "content": CRITIC_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Papers:\n{build_context(papers)}\n\n"
                f"Current answer:\n{answer}"
            ),
        },
    ]
    return llm.chat_json(messages, max_tokens=512)


# ---------------------------------------------------------------- nodes

def node_explore_scope(state):
    """One exploration turn: propose the next layer of directions."""
    action = (state.get("explore_action") or "scope").strip()
    parent = state.get("explore_parent") or {}
    path = list(state.get("explore_path") or [])
    seen = list(state.get("explore_seen") or [])
    if action == "deepen" and parent:
        _emit(f"🔭 深入方向：{parent.get('name', '')}（展开下一层）")
        directions = explore.deepen(parent, path=path, seen=seen)
    else:
        _emit(f"🔭 领域拆解：把「{state['question'][:40]}」切成可分别调研的方向")
        directions = explore.scope(state["question"], path=path, seen=seen)
    meta = {"action": action, "count": len(directions), "degraded": False}
    if not directions:
        # a model hiccup must not dead-end the session: hand back one direction
        # that is just the user's own topic, so "生成综述报告" still works.
        _emit("🔭 方向拆解失败：退化为单一方向（直接用你的问题检索）")
        directions = [{"id": "1", "name": state["question"][:20] or "本次主题",
                       "en": state["question"][:80], "why": "（拆解失败，直接检索该主题）",
                       "queries": [state["question"][:160]], "parent": "",
                       "depth": 0, "refs": []}]
        meta["degraded"] = True
    if state.get("explore_preview", True):
        _emit(f"🔭 代表作预览：按被引量为 {len(directions)} 个方向各取 "
              f"{explore.PREVIEW_PER_DIRECTION} 篇")
        try:
            explore.preview(directions)
        except Exception as e:
            meta["preview_error"] = f"{type(e).__name__}: {e}"
    meta["with_refs"] = sum(1 for d in directions if d.get("refs"))
    return {"explore_directions": directions, "explore_meta": meta}


def node_survey_plan(state):
    """Chosen directions -> the queries the ordinary retrieval chain runs."""
    directions = state.get("directions") or state.get("explore_directions") or []
    queries = explore.survey_queries(directions)
    if not queries:
        queries = [state["question"]]
    _emit(f"🔭 综述检索规划：{len(directions)} 个方向 → {len(queries)} 条检索词")
    print(f"      [survey] directions: {explore.names(directions)}")
    print(f"      [survey] queries: {queries}")
    return {"queries": queries, "directions": directions}


def survey_target(state):
    """What the reranker/critic should score against for a review run: the
    topic *plus* the chosen directions, so a section's papers are not pushed
    out by papers that merely share the topic's words."""
    base = state.get("question") or ""
    names = [d.get("name") or d.get("en") or "" for d in (state.get("directions") or [])]
    names = [n for n in names if n]
    return base if not names else base + " | " + "; ".join(names)


def node_plan_queries(state):
    _emit("① 查询规划：DeepSeek 将问题拆成 2-3 条英文检索词…")
    return {"queries": plan_queries(state["question"])}


def node_retrieve(state):
    _emit(f"② 第 {state['rounds'] + 1} 轮检索：{len(search.available_sources())} 个数据源并行")
    queries = state["queries"]
    label = queries[0] if len(queries) == 1 else f"{len(queries)} follow-up queries"
    print(f"[round {state['rounds'] + 1}] Retrieving for: {label} "
          f"(sources: {', '.join(search.available_sources())})")
    pool = dict(state["pool"])
    added, queries_log = add_papers_many(
        pool, queries, limit=PAPERS_PER_QUERY, filters=state.get("filters")
    )
    classic_added = 0
    if state["rounds"] == 0:
        # the two retrieval passes are the longest single steps in this node:
        # check between them so a stop lands in seconds, not at the node's end
        _check_cancel()
        # Relevance ranking is dominated by whatever is new; one citation-ranked
        # pass per run drags the field's anchor papers into the pool.
        anchors = queries[:CLASSIC_QUERIES]
        _emit(f"②b 经典锚点检索：按被引量排序 {len(anchors)} 条检索词，捞回高被引奠基工作")
        classic_added, classic_log = add_papers_many(
            pool, anchors, limit=search.CLASSIC_PER_QUERY,
            filters=state.get("filters"), classic=True,
        )
        queries_log += classic_log
        print(f"      [classic] {classic_added} landmark paper(s) added "
              f"(>= {search.CLASSIC_MIN_CITATIONS} citations, else the top hit per query)")
    if added + classic_added == 0:
        # A run where every source errored is not "no papers match": on
        # 2026-10-19 the machine's outbound proxy was down, all six sources
        # answered "Connection refused", and the job still ended "✅ 完成"
        # with an empty list -- the user had no way to tell a dead network
        # from an empty literature. Only an answer from *some* source can
        # honestly produce "no results", so raise and let the job fail with
        # the reason.
        errors = [q.get("error") for q in queries_log if q.get("error")]
        if errors and all(q.get("error") for q in queries_log):
            raise RuntimeError(
                "所有检索数据源都请求失败（这不是「没有相关文献」）："
                + errors[0][:400]
                + "\n  提示：先确认本机能访问外网（直连通常不通，需要代理），"
                  "再检查 .env 里的 HTTP_PROXY / HTTPS_PROXY 是否可用。"
            )
    tier = journal.current_filter()
    if tier is not None and tier.enabled:
        summary = tier.summary()
        print(f"      [tier] 刊物层级（{'、'.join(summary['labels'])}）只看已见的论文："
              f"保留 {summary['kept']} 篇，滤除 {summary['dropped']} 篇"
              f"（其中 {summary['dropped_unclassified']} 篇没有期刊层级数据）")
    log = list(state.get("retrieval_log") or [])
    log.append({
        "round": state["rounds"] + 1,
        "queries": queries_log,
        "added": added + classic_added,
    })
    return {"pool": pool, "rounds": state["rounds"] + 1, "retrieval_log": log}


def node_rerank(state):
    pool_papers = list(state["pool"].values())
    if not pool_papers:
        _emit("③ 排序：论文池为空")
        print("      No papers with abstracts found.")
        return {
            "papers": [],
            "scores": [],
            "answer": "No papers with abstracts found. Try rephrasing the question.",
        }
    sort = state.get("sort", "relevance")
    # a review covers several directions at once: score against the topic plus
    # the chosen directions, and keep a wider working set than a single question
    survey = state.get("mode") == "survey"
    target = survey_target(state) if survey else state["question"]
    cap = SURVEY_MAX_PAPERS if survey else MAX_TOTAL_PAPERS
    if state.get("mode") == "search":
        # search-only mode: no cap, the whole deduped pool is returned
        if sort != "relevance":
            _emit("③ 排序：按所选字段对全量论文池排序（不限上限）")
            papers = sort_papers(pool_papers, sort)
            print(f"      [sort] pool {len(pool_papers)} -> all {len(papers)} by {sort}")
            return {"papers": papers, "scores": []}
        _emit("③ 相关性重排：对全量论文池打分（不限上限）")
        reranked = rerank.rerank_papers(target, pool_papers, top_k=None)
        if reranked:
            papers = [p for p, _ in reranked]
            scores = [{"title": p["title"], "score": round(s, 2)} for p, s in reranked]
            top3 = "; ".join(f"{p['title'][:38]}({s:.1f})" for p, s in reranked[:3])
            print(f"      [rerank] pool {len(pool_papers)} -> all {len(papers)}: {top3}")
        else:
            papers = pool_papers
            scores = []
            print(f"      [rerank] scoring failed; keeping all {len(papers)} papers")
        return {"papers": papers, "scores": scores}
    _emit("③ 重排：DeepSeek 按与问题的相关性给论文池打分")
    reranked = rerank.rerank_papers(target, pool_papers, top_k=cap)
    if reranked:
        papers = [p for p, _ in reranked]
        scores = [{"title": p["title"], "score": round(s, 2)} for p, s in reranked]
        top3 = "; ".join(f"{p['title'][:38]}({s:.1f})" for p, s in reranked[:3])
        print(f"      [rerank] pool {len(pool_papers)} -> top {len(papers)}: {top3}")
    else:
        papers = pool_papers[:cap]
        scores = []
    print(f"      Pool now has {len(papers)} papers")
    return {"papers": papers, "scores": scores}


def node_screen_retractions(state):
    papers = state.get("papers") or []
    if not papers:
        return {"retraction": {"total": 0, "checked": 0, "retracted": 0, "unchecked": 0}}
    if not state.get("screen_retractions", True):
        _emit("④ 撤稿筛查：已跳过（仅检索模式，可在界面勾选开启）")
        print("      Retraction screening skipped (search-only mode).")
        return {"retraction": {"total": len(papers), "checked": 0, "retracted": 0,
                               "unchecked": len(papers), "skipped": True,
                               "reason": "仅检索论文模式已跳过撤稿筛查"}}
    _emit(f"④ 撤稿筛查：核对 {len(papers)} 篇论文是否已被撤稿")
    try:
        summary = retraction.screen(papers)
    except Exception as e:
        print(f"      [warn] retraction screening failed: {e}")
        summary = {"total": len(papers), "checked": 0, "retracted": 0,
                   "unchecked": len(papers), "error": str(e)}
    if summary.get("retracted"):
        print(f"      [retraction] {summary['retracted']} retracted paper(s) in the set")
    return {"retraction": summary}


def node_score_evidence(state):
    """Evidence scoring is an *ask-mode* feature: it exists to help judge the
    evidence behind a written answer. Search mode is deliberately left as a
    plain, fast retrieval listing."""
    papers = state.get("papers") or []
    if state.get("mode") not in ("ask", "survey"):
        return {"evidence": {"skipped": True, "reason": "仅智能问答 / 综述报告模式支持证据评分"}}
    if not state.get("score_evidence"):
        return {"evidence": {"skipped": True}}
    if not papers:
        return {"evidence": {"scored": 0, "unscored": 0, "mean": None, "grades": {}}}
    weights = state.get("weights") or scoring.load_settings()["weights"]
    target_species = state.get("target_species") or ""
    _emit(f"④b 证据评分：{len(papers)} 篇 × {len(scoring.DIMENSIONS)} 个维度"
          + (f"（目标物种：{target_species}）" if target_species else ""))
    try:
        scored, summary = scoring.score_papers(
            survey_target(state) if state.get("mode") == "survey" else state["question"],
            papers, weights, target_species,
            on_progress=lambda done, total: print(f"      [scoring] {done}/{total} papers scored"),
        )
    except Exception as e:
        print(f"      [warn] evidence scoring failed: {e}")
        return {"evidence": {"error": str(e), "scored": 0, "unscored": len(papers), "grades": {}}}
    print("      [scoring] " + scoring.summarize_label(summary))
    return {"papers": scored, "evidence": summary}


def node_generate(state):
    survey = state.get("mode") == "survey"
    if survey:
        sections = explore.outline(state.get("directions") or [])
        _emit(f"⑤ 综述写作：按 {len(state.get('directions') or [])} 个方向写带引用的报告"
              f"（{len(state.get('papers') or [])} 篇论文）")
        print("      Writing the survey report...")
        if sections:
            print(f"      [survey] sections:\n{sections}")
        messages = [
            {"role": "system", "content": SURVEY_PROMPT},
            {
                "role": "user",
                "content": (f"【调研主题】{state['question']}\n\n"
                            f"【报告目录】\n{sections or '（未指定，按主题自行划分）'}\n\n"
                            f"【论文表】\n{build_context(state['papers'])}"),
            },
        ]
        try:
            return {"answer": llm.chat(messages, max_tokens=6000)}
        except Exception as e:
            print(f"      Survey writing failed ({e}); falling back to the paper list.")
            _emit("⑤ 综述写作失败：退化为论文列表输出")
            listing = format_papers(state["papers"], "相关性")
            return {"answer": f"{listing}\n\n---\n\n> ⚠️ 综述写作阶段失败"
                               f"（{type(e).__name__}: {e}），以上为本次检索到的论文列表。"}
    _emit(f"⑤ 生成回答：基于 top {len(state.get('papers') or [])} 篇论文写带引用的综述")
    print(f"      Generating answer (round {state['rounds']})...")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {state['question']}\n\nPapers:\n{build_context(state['papers'])}",
        },
    ]
    try:
        return {"answer": llm.chat(messages)}
    except Exception as e:
        # every other network-dependent node degrades gracefully; this one used
        # to raise, which threw away a whole retrieval (and the wait for it) --
        # hand back the paper list instead so the run is still worth something.
        print(f"      Generation failed ({e}); falling back to the paper list.")
        _emit("⑤ 生成回答失败：退化为论文列表输出")
        listing = format_papers(state["papers"], "相关性")
        return {"answer": f"{listing}\n\n---\n\n> ⚠️ 生成阶段失败"
                           f"（{type(e).__name__}: {e}），以上为本次检索到的论文列表。"}


def node_format_papers(state):
    _emit("✅ 检索完成：整理论文列表")
    label = SEARCH_SORT_LABELS.get(state.get("sort", "relevance"), "相关性")
    return {"answer": format_papers(state["papers"], label)}


def node_critique(state):
    _emit("⑥ 证据充分性评估：是否需要追加检索")
    print("      Critiquing evidence sufficiency...")
    try:
        verdict = criticize(
            survey_target(state) if state.get("mode") == "survey" else state["question"],
            state["papers"], state["answer"])
    except Exception as e:
        print(f"      Critique failed ({e}); stopping iteration.")
        verdict = {"sufficient": True, "follow_up_queries": []}
    follow_ups = [q for q in verdict.get("follow_up_queries", []) if q.strip()]
    if verdict.get("sufficient"):
        print("      Evidence judged sufficient.")
    elif follow_ups:
        print(f"      Follow-up queries: {follow_ups}")
    else:
        print("      No follow-up queries proposed; stopping.")
    return {"verdict": verdict, "queries": follow_ups}


def node_verify_rules(state):
    _emit("⑦ 三层校验 1/3：引用规则校验")
    try:
        # `directions` (survey runs) switches on the report-shape checks: the
        # rule layer is the only layer that knows what the report was supposed
        # to contain, because the plan is still in state.
        audit = verify.audit_rules(state["answer"], state["papers"],
                                   directions=state.get("directions"))
    except Exception as e:
        audit = {
            "lines": ["--- Citation verification ---", f"BAD rule check failed: {e}"],
            "items": [{"status": "bad", "message": str(e)}],
        }
    return {"reports": [audit["lines"]], "rule_audit": audit}


def node_verify_llm(state):
    _emit("⑧ 三层校验 2/3：LLM 归因（每条引用是否被该论文支持）")
    try:
        audit = verify.llm_audit(state["answer"], state["papers"])
    except Exception as e:
        audit = {
            "lines": ["--- LLM citation verification (DeepSeek) ---", f"BAD LLM check failed: {e}"],
            "verdicts": [],
        }
    return {"reports": [audit["lines"]], "llm_audit": audit}


def node_verify_sentences(state):
    if state.get("deep_verify", True) is False:
        _emit("⑨ 三层校验 3/3：逐句全文校验已跳过（快速模式）")
        audit = {
            "lines": ["--- Per-sentence evidence check ---",
                      "SKIP per-sentence full-text check disabled for this run"],
            "verdicts": [],
            "passages_used": 0,
            "parse_failures": [],
            "skipped": True,
        }
        return {"reports": [audit["lines"]], "claim_audit": audit}
    _emit("⑨ 三层校验 3/3：逐句证据（全文检索，可能较慢）")
    # work on a copy: the evidence search widens the pool, and that widening is
    # handed back through the return value (a state dict mutated in place is
    # invisible to LangGraph, so a resumed run would see the stale pool)
    pool = dict(state.get("pool") or {})
    try:
        # state["papers"] is exactly the list the answer's [n] markers refer to;
        # the pool is only extra material for the evidence search.
        audit = sentences.run(state["answer"], state["papers"], pool,
                              should_cancel=_check_cancel)
    except JobCancelled:
        raise       # a stop is not a check failure: let it unwind the graph
    except Exception as e:
        audit = {
            "lines": ["--- Per-sentence evidence check ---", f"BAD per-sentence check failed: {e}"],
            "verdicts": [],
            "passages_used": 0,
            "parse_failures": [str(e)],
        }
    widened = audit.pop("pool", None)
    out = {"reports": [audit["lines"]], "claim_audit": audit}
    if isinstance(widened, dict):
        out["pool"] = widened
    return out


# ---------------------------------------------------------------- routing

def node_chat_answer(state):
    """Answer a follow-up inside the finished run's working set."""
    message = (state.get("chat_message") or "").strip()
    papers = state.get("papers") or []
    _emit(f"💬 追问回答：基于本次 {len(papers)} 篇论文作答（不重新检索）")
    print(f"      Answering follow-up: {message[:80]!r}")
    fresh_from = len(papers) - len(state.get("chat_new_papers") or []) + 1 \
        if state.get("chat_new_papers") else 0
    out = chat.reply(state.get("question", ""), state.get("answer", ""), papers,
                     state.get("chat_history") or [], message, fresh_from=fresh_from)
    rounds = int(state.get("chat_rounds") or 0)
    if out.get("needs_new_search") and state.get("chat_allow_retrieve", True) \
            and rounds < CHAT_MAX_ROUNDS:
        _emit(f"💬 追问回答：现有材料不足，准备补检索（{out.get('search_query') or '—'}）")
    elif out.get("needs_new_search"):
        _emit("💬 追问回答完成：现有材料不足，已给出补检索建议")
    else:
        _emit("💬 追问回答完成" + (f"（已用上本轮新补的 {len(state.get('chat_new_papers') or [])} 篇）"
                                  if rounds else ""))
    return {
        "chat_answer": out["answer"],
        "chat_meta": {
            "needs_new_search": bool(out.get("needs_new_search")),
            "search_query": out.get("search_query") or "",
            "json_parsed": bool(out.get("json_parsed")),
            "model_seconds": out.get("duration_s"),
            "rounds": rounds,
        },
    }


def node_chat_retrieve(state):
    """Give the follow-up the material it said it was missing.

    Papers are *appended*. The earlier answer's `[3]` must keep meaning the same
    paper, so nothing already in the working set is re-ranked away; new papers
    simply continue the numbering. The batch is relevance-scored against the
    follow-up itself, so a vague query cannot flood the list.
    """
    meta = state.get("chat_meta") or {}
    query = (meta.get("search_query") or "").strip()
    papers = list(state.get("papers") or [])
    rounds = int(state.get("chat_rounds") or 0) + 1
    tried = list(state.get("chat_tried") or [])
    if not query:
        return {"chat_rounds": rounds}
    _emit(f"🔎 追问补检索（第 {rounds} 次）：{query[:60]}")
    print(f"      Follow-up retrieval for: {query!r}")
    pool = dict(state.get("pool") or {})
    have = {verify.normalize(p.get("title") or "") for p in papers}
    before = set(pool)
    try:
        add_papers_many(pool, [query], limit=PAPERS_PER_QUERY,
                        filters=state.get("filters"))
    except Exception as e:
        print(f"      Follow-up retrieval failed ({e})")
        return {"chat_rounds": rounds, "chat_tried": tried + [query.lower()]}
    fresh = [pool[k] for k in pool if k not in before and k not in have]
    if fresh:
        try:
            ranked = rerank.rerank_papers(message_or_question(state), fresh,
                                          top_k=CHAT_MAX_NEW_PAPERS)
        except Exception as e:
            print(f"      Follow-up rerank failed ({e}); keeping retrieval order")
            ranked = None
        if ranked:
            fresh = [p for p, _ in ranked]
    room = max(0, CHAT_MAX_TOTAL_PAPERS - len(papers))
    added = fresh[:min(CHAT_MAX_NEW_PAPERS, room)]
    for i, paper in enumerate(added, len(papers) + 1):
        print(f"      [{i}] {paper['title'][:70]}")
    if not added:
        _emit("🔎 追问补检索：没有找到新的相关论文")
    else:
        _emit(f"🔎 追问补检索：新增 {len(added)} 篇（编号 [{len(papers) + 1}]"
              f"{'-' if len(added) > 1 else ''}[{len(papers) + len(added)}]）")
    return {
        "papers": papers + added,
        "pool": pool,
        "chat_new_papers": list(state.get("chat_new_papers") or []) + added,
        "chat_rounds": rounds,
        "chat_tried": tried + [query.lower()],
        "chat_log": list(state.get("chat_log") or []) + [{
            "query": query, "added": len(added),
            "titles": [p["title"] for p in added],
        }],
    }


def message_or_question(state):
    """What a follow-up's retrieval should be scored against: the follow-up
    itself, anchored by the original question (a bare "再说说这个" carries no
    retrieval signal on its own)."""
    message = (state.get("chat_message") or "").strip()
    question = (state.get("question") or "").strip()
    return f"{question} | {message}" if question and message else (message or question)


def route_after_chat(state):
    """After a follow-up answer: fetch what it is missing, or go verify it."""
    meta = state.get("chat_meta") or {}
    if not meta.get("needs_new_search"):
        return "chat_verify"
    if not state.get("chat_allow_retrieve", True):
        return "chat_verify"
    if int(state.get("chat_rounds") or 0) >= CHAT_MAX_ROUNDS:
        return "chat_verify"
    query = (meta.get("search_query") or "").strip().lower()
    if not query or query in [q.lower() for q in (state.get("chat_tried") or [])]:
        return "chat_verify"
    return "chat_retrieve"


def node_chat_verify(state):
    """Rule layer only: a follow-up turn is cheap, and this is the layer that
    catches a citation number the model invented."""
    _emit("💬 追问校验：引用规则校验")
    try:
        audit = verify.audit_rules(state.get("chat_answer") or "",
                                   state.get("papers") or [])
    except Exception as e:
        audit = {"lines": ["--- Citation verification ---",
                           f"BAD rule check failed: {e}"],
                 "items": [{"status": "bad", "message": str(e)}]}
    return {"chat_audit": audit}


def route_entry(state):
    """Which branch this invocation is.

    Three entry points share one compiled graph: an exploration turn (propose
    directions), a survey run (a review over the chosen directions, which then
    walks the ordinary chain), and everything else -- a follow-up question goes
    to the chat branch, an ordinary run starts from query planning.
    """
    action = (state.get("explore_action") or "").strip()
    if action in ("scope", "deepen"):
        return "explore_scope"
    if action == "survey":
        return "survey_plan"
    return "chat_answer" if (state.get("chat_message") or "").strip() else "plan_queries"


def route_after_rerank(state):
    """After rerank (and again after score_evidence, which keeps the order):
    search mode lists the papers, ask mode writes an answer."""
    if not state["papers"]:
        return "end"
    return "format_papers" if state.get("mode") == "search" else "generate"


def route_after_generate(state):
    if state["rounds"] >= MAX_ROUNDS:
        print("      Max rounds reached; stopping.")
        return "verify"
    return "critique"


def route_after_critique(state):
    if (state.get("verdict") or {}).get("sufficient"):
        return "verify"
    if not state.get("queries"):
        return "verify"
    return "retrieve"


# ---------------------------------------------------------------- graph

def _timed(name, fn):
    """Wrap a node so every stage reports its wall-clock cost."""

    def wrapper(state):
        _check_cancel()
        t0 = time.monotonic()
        out = fn(state)
        dt = time.monotonic() - t0
        print(f"      [time] {name}: {dt:.1f}s")
        hook = _hooks_for()[1]
        if hook:
            try:
                hook(name, round(dt, 2))
            except Exception:
                pass
        if isinstance(out, dict):
            out = {**out, "timings": {name: round(dt, 2)}}
        return out

    return wrapper


def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("explore_scope", _timed("explore_scope", node_explore_scope))
    builder.add_node("survey_plan", _timed("survey_plan", node_survey_plan))
    builder.add_node("plan_queries", _timed("plan_queries", node_plan_queries))
    builder.add_node("retrieve", _timed("retrieve", node_retrieve))
    builder.add_node("rerank", _timed("rerank", node_rerank))
    builder.add_node("screen_retractions", _timed("screen_retractions", node_screen_retractions))
    builder.add_node("score_evidence", _timed("score_evidence", node_score_evidence))
    builder.add_node("generate", _timed("generate", node_generate))
    builder.add_node("format_papers", _timed("format_papers", node_format_papers))
    builder.add_node("critique", _timed("critique", node_critique))
    builder.add_node("verify_rules", _timed("verify_rules", node_verify_rules))
    builder.add_node("verify_llm", _timed("verify_llm", node_verify_llm))
    builder.add_node("verify_sentences", _timed("verify_sentences", node_verify_sentences))
    builder.add_node("chat_answer", _timed("chat_answer", node_chat_answer))
    builder.add_node("chat_retrieve", _timed("chat_retrieve", node_chat_retrieve))
    builder.add_node("chat_verify", _timed("chat_verify", node_chat_verify))

    builder.add_conditional_edges(
        START,
        route_entry,
        {"chat_answer": "chat_answer", "plan_queries": "plan_queries",
         "explore_scope": "explore_scope", "survey_plan": "survey_plan"},
    )
    # an exploration turn ends by itself: the user has to choose a direction
    # before anything is retrieved.
    builder.add_edge("explore_scope", END)
    # a survey reuses the whole ordinary chain -- retrieval, screening,
    # scoring, writing, three-layer verification -- with a review prompt.
    builder.add_edge("survey_plan", "retrieve")
    builder.add_conditional_edges(
        "chat_answer",
        route_after_chat,
        {"chat_retrieve": "chat_retrieve", "chat_verify": "chat_verify"},
    )
    # the second pass answers with the fetched papers already in the list
    builder.add_edge("chat_retrieve", "chat_answer")
    builder.add_edge("chat_verify", END)
    builder.add_edge("plan_queries", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "screen_retractions")
    builder.add_edge("screen_retractions", "score_evidence")
    builder.add_conditional_edges(
        "score_evidence",
        route_after_rerank,
        {"generate": "generate", "format_papers": "format_papers", "end": END},
    )
    builder.add_edge("format_papers", END)
    builder.add_conditional_edges(
        "generate",
        route_after_generate,
        {"critique": "critique", "verify": "verify_rules"},
    )
    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {"retrieve": "retrieve", "verify": "verify_rules"},
    )
    builder.add_edge("verify_rules", "verify_llm")
    builder.add_edge("verify_llm", "verify_sentences")
    builder.add_edge("verify_sentences", END)
    # a checkpointer makes every super-step resumable; None keeps it stateless
    return builder.compile(checkpointer=_checkpointer)


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def _clean_filters(filters):
    out = {"min_year": None, "max_year": None, "review_only": False, "venue": "",
           "tiers": [], "keep_others": False}
    for key in ("min_year", "max_year"):
        try:
            out[key] = int((filters or {}).get(key))
        except (TypeError, ValueError):
            out[key] = None
    out["review_only"] = bool((filters or {}).get("review_only"))
    out["venue"] = str((filters or {}).get("venue") or "").strip()
    # 正刊/子刊/一区-四区: keep only the keys this build knows, so an unknown box
    # in an old request (or a typo) cannot silently disable the filter
    tiers = (filters or {}).get("tiers") or []
    out["tiers"] = [t for t in tiers if t in journal.TIER_KEYS]
    out["keep_others"] = bool((filters or {}).get("keep_others"))
    return out


def run_full(question, mode="ask", sort="relevance", filters=None,
             on_stage=None, deep_verify=True, on_timing=None,
             score_evidence=False, weights=None, target_species="",
             screen_retractions=None, thread_id=None, resume=False,
             directions=None):
    """Execute the literature pipeline graph and return the full result state
    (answer, papers, audits, retrieval_log, ...) for snapshots and the UI.

    `screen_retractions=None` keeps the mode default: always on for an answer,
    off for a search-only listing unless RETRACTION_IN_SEARCH is set. Pass True
    to screen anyway (the UI checkbox does).

    `thread_id` names the checkpoint thread (any string; the web layer uses the
    job id). `resume=True` continues that thread from its last completed node
    instead of starting over -- used after a failed run.
    """
    if sort not in SEARCH_SORTS:
        sort = "relevance"
    filters = _clean_filters(filters)
    if mode == "survey" and not directions:
        raise ValueError("survey mode needs the chosen directions")
    if screen_retractions is None:
        screen_retractions = (mode in ("ask", "survey")) or RETRACTION_IN_SEARCH
    screen_retractions = bool(screen_retractions)
    initial = {
        "question": question,
        "mode": mode,
        "sort": sort,
        "filters": filters,
        "queries": [],
        "pool": {},
        "papers": [],
        "answer": "",
        "retraction": {"total": 0, "checked": 0, "retracted": 0, "unchecked": 0},
        "screen_retractions": screen_retractions,
        "rounds": 0,
        "verdict": {},
        "retrieval_log": [],
        "scores": [],
        "rule_audit": {"lines": [], "items": []},
        "llm_audit": {"lines": [], "verdicts": []},
        "claim_audit": {
            "lines": [], "verdicts": [], "passages_used": 0, "parse_failures": [],
        },
        "deep_verify": deep_verify,
        "score_evidence": bool(score_evidence),
        "weights": weights or {},
        "target_species": target_species or "",
        "evidence": {},
        "reports": [],
        "timings": {},
        "explore_action": "",
        "explore_directions": [],
        "directions": list(directions or []),
        "explore_preview": False,
        "explore_meta": {},
        "explore_path": [],
        "explore_seen": [],
    }
    timings = {}

    def _collect(name, seconds):
        timings[name] = timings.get(name, 0.0) + seconds
        if on_timing:
            on_timing(name, seconds)

    if not thread_id:
        thread_id = "run-" + uuid.uuid4().hex[:12] if CHECKPOINT_ENABLED else None
    if resume and not has_checkpoint(thread_id):
        raise ValueError(f"没有可续跑的断点：{thread_id}")
    install_hooks(on_stage, _collect)
    # the venue-class filter is run-scoped: two runs in one process have their
    # own boxes and their own tally, while the journal table behind them (and
    # what the other run already looked up) is shared
    tier = journal.TierFilter(filters.get("tiers"), filters.get("keep_others"))
    tier_token = journal.set_filter(tier)
    graph_app = get_app()
    try:
        # the critique -> retrieve loop makes the longest path a function of
        # MAX_ROUNDS (22 node visits at the current 3); LangGraph's default
        # limit is 25, so it is set explicitly rather than left near the edge
        if resume:
            # None tells LangGraph to continue the thread's pending tasks
            result = graph_app.invoke(None, config=_config(thread_id))
        else:
            result = graph_app.invoke(initial, config=_config(thread_id))
    finally:
        clear_hooks()
        journal.reset_filter(tier_token)
    # a resumed run only measures the stages it re-ran: add them to what the
    # earlier attempt already spent, so the UI shows the real total
    result["tier_filter"] = tier.summary()
    result["timings"] = dict(result.get("timings") or {})
    result["thread_id"] = thread_id
    return result


def run_explore(action="scope", topic="", parent=None, path=None, seen=None,
                preview=True, on_stage=None, on_timing=None, thread_id=None):
    """One exploration turn through the same compiled graph.

    `action="scope"` splits the topic; `action="deepen"` splits one chosen
    direction. The turn carries its own context (path/seen), so a session spans
    as many invocations as the user wants without any checkpoint dependence.
    """
    initial = {
        "question": topic or "",
        "mode": "explore",
        "queries": [],
        "pool": {},
        "papers": [],
        "answer": "",
        "rounds": 0,
        "reports": [],
        "timings": {},
        "explore_action": action,
        "explore_parent": dict(parent or {}),
        "explore_path": list(path or []),
        "explore_seen": list(seen or []),
        "explore_preview": bool(preview),
        "directions": [],
    }
    if not thread_id:
        thread_id = "explore-" + uuid.uuid4().hex[:12] if CHECKPOINT_ENABLED else None
    install_hooks(on_stage, on_timing)
    try:
        result = get_app().invoke(initial, config=_config(thread_id))
    finally:
        clear_hooks()
    result["thread_id"] = thread_id
    return result


def run_chat(message, question="", answer="", papers=None, history=None,
             on_stage=None, on_timing=None, thread_id=None,
             allow_retrieve=True, filters=None, pool=None):
    """One interactive follow-up turn through the same compiled graph.

    Everything the turn needs is handed in as state, so it does not depend on
    the original run's checkpoint still being in memory: the working set is the
    same list the answer cited, which is what keeps `[n]` stable.
    """
    initial = {
        "question": question or "",
        "answer": answer or "",
        "papers": list(papers or []),
        "chat_message": message or "",
        "chat_history": list(history or []),
        "chat_allow_retrieve": bool(allow_retrieve),
        "chat_rounds": 0,
        "chat_tried": [],
        "chat_new_papers": [],
        "chat_log": [],
        "filters": _clean_filters(filters),
        "mode": "chat",
        "queries": [],
        "pool": dict(pool or {}),
        "rounds": 0,
        "reports": [],
        "timings": {},
    }
    if not thread_id:
        thread_id = "chat-" + uuid.uuid4().hex[:12] if CHECKPOINT_ENABLED else None
    install_hooks(on_stage, on_timing)
    tier = journal.TierFilter(initial["filters"].get("tiers"),
                              initial["filters"].get("keep_others"))
    tier_token = journal.set_filter(tier)
    try:
        result = get_app().invoke(initial, config=_config(thread_id))
    finally:
        clear_hooks()
        journal.reset_filter(tier_token)
    result["tier_filter"] = tier.summary()
    result["thread_id"] = thread_id
    return result


def resume_full(thread_id, on_stage=None, on_timing=None):
    """Continue a failed run from its last checkpoint (no re-retrieval).

    Raises ValueError when the thread has nothing to resume."""
    return run_full("", thread_id=thread_id, resume=True,
                    on_stage=on_stage, on_timing=on_timing)


def run(question, mode="ask", sort="relevance", filters=None):
    """CLI/legacy wrapper around run_full; prints the verification reports."""
    result = run_full(question, mode=mode, sort=sort, filters=filters)
    for block in result.get("reports", []):
        print()
        print("\n".join(block))
    return result.get("answer") or "[error] no answer produced"


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:])
    if not question:
        sys.exit('Usage: python3 graph.py "your research question"')
    print()
    print(run(question))
