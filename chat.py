"""Follow-up turns on a finished answer (pure stdlib).

A completed run leaves a working set of papers and a written answer. The user
almost always has a second question -- "which of these actually did the
experiment?", "把这些结论整理成一段综述" -- and re-running the whole retrieval
for it is both slow and wrong: the citation numbers would move.

This module answers inside the *existing* working set. The paper list is handed
to the model exactly as the answer cited it, so `[3]` still means the same
paper, and the honest answer to "摘要里没写" is `needs_new_search`, which the UI
offers as a one-click re-search instead of a fabricated number.
"""

import time

import llm

MAX_ABSTRACT_CHARS = int(1400)
MAX_ANSWER_CHARS = int(9000)
MAX_HISTORY_TURNS = int(6)
MAX_HISTORY_CHARS = int(1200)

SYSTEM_PROMPT = """你是 ScholarMate 的科研助手，正在与用户围绕**一次已经完成的文献调研**对话。\
用户会追问细节，你基于下面给出的材料回答。

硬性规则：
1. 只能使用【本次工作论文集】与【上一步的回答】里出现的信息；不得引入未列出的文献，\
不得编造数据、样本量、实验条件、结论强度。
2. 引用必须沿用论文表的编号写作 [1]、[3]；不要创造新编号，也不要引用表以外的文献。正文用到引用时，结尾必须附一节 `References:`，逐条写成 `[1] 标题 (年份)`，只列本轮真正用到的论文（这正是规则层校验的依据；没有引用就不写这一节）。
3. 如果材料不足以回答（例如问的是没检索到的话题，或需要摘要里没有的细节），\
把 needs_new_search 设为 true，并在 search_query 里给出 1 条可直接检索的英文关键词；\
**系统会立刻用它去补检索，再把新论文追加进上面的论文表重新问你一遍**，所以这一轮的 answer \
只需一两句话说明你还缺什么（会被下一轮覆盖，不必展开）；如果已经补过一次仍未解决，\
就照实说明检索没能覆盖，不要用模糊措辞假装回答了。
4. 用中文回答，直接、紧凑、面向科研写作；可用 Markdown 的小标题、列表、表格。\
不要复述整篇原回答，只回答用户这一问。
5. 涉及证据强弱比较时，只依据材料里给出的证据评分、全文/摘要来源与结论，不臆断影响力。

只输出 JSON：{"answer": "Markdown 正文", "needs_new_search": false, "search_query": ""}"""


def _clip(text, limit):
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " …（截断）"


def build_context(papers):
    """The working set, numbered exactly as the answer cited it."""
    blocks = []
    for i, p in enumerate(papers or [], 1):
        authors = "; ".join((p.get("authors") or [])[:3])
        lines = [
            f"[{i}] {p.get('title') or ''} ({p.get('year') or '年份未知'})",
            f"Authors: {authors}" if authors else "",
            f"Venue: {p.get('venue')}" if p.get("venue") else "",
            f"Citations: {p.get('citations')}" if p.get("citations") is not None else "",
            f"URL: {p.get('url')}" if p.get("url") else "",
        ]
        ev = p.get("evidence_score") or (p.get("evidence") or {}).get("total")
        if ev is not None:
            grade = (p.get("evidence") or {}).get("grade") or ""
            lines.append(f"Evidence score: {ev}/100 {grade}".rstrip())
        if p.get("retracted"):
            lines.append("Status: *** 已撤稿 *** 不可作为证据")
        lines.append("Abstract: " + (_clip(p.get("abstract"), MAX_ABSTRACT_CHARS)
                                    or "(无摘要)"))
        blocks.append("\n".join(x for x in lines if x))
    return "\n\n".join(blocks) or "(本次没有论文)"


def build_prompt(question, answer, papers, history, message, fresh_from=0):
    """`fresh_from` is the 1-based index of the first paper this turn fetched:
    the model is told which papers are new, so it does not treat them as
    something the user has already seen answered."""
    parts = [f"【本次工作论文集】（共 {len(papers or [])} 篇，编号与回答中的引用一致）",
             build_context(papers), ""]
    if answer:
        parts += ["【上一步的回答】", _clip(answer, MAX_ANSWER_CHARS), ""]
    turns = [h for h in (history or []) if (h.get("content") or "").strip()]
    if turns:
        parts.append("【对话历史】（较早的在前）")
        for h in turns[-MAX_HISTORY_TURNS:]:
            who = "用户" if h.get("role") == "user" else "ScholarMate"
            line = _clip(h.get("content"), MAX_HISTORY_CHARS)
            parts.append(f"{who}：{line}")
        parts.append("")
    parts += [f"【本次问题】{question or '（未记录，见上一步回答）'}",
              "",
              f"【用户的新问题】{(message or '').strip()}",
              "",
              "请按系统提示只输出 JSON。"]
    if fresh_from:
        parts += ["", f"（注意：[{fresh_from}] 起是**这一轮刚刚补检索到的论文**，"
                      f"用户此前没有见过，回答时可以明确说这是新检索到的证据。）"]
    return "\n".join(parts)


def reply(question, answer, papers, history, message, max_tokens=2000, fresh_from=0):
    """One grounded follow-up turn.

    Returns {"answer", "needs_new_search", "search_query", "json_parsed",
    "duration_s"}. Raises when the model is unreachable -- the caller turns
    that into a visible error bubble rather than a silent empty reply.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": build_prompt(question, answer, papers, history, message,
                                 fresh_from=fresh_from)},
    ]
    t0 = time.monotonic()
    try:
        out = llm.chat_json(messages, temperature=0.35, max_tokens=max_tokens)
        if not isinstance(out, dict):
            raise ValueError("model did not return an object")
        text = str(out.get("answer") or "").strip()
        if not text:
            raise ValueError("empty answer field")
        parsed = True
        needs = bool(out.get("needs_new_search"))
        query = str(out.get("search_query") or "").strip()
    except ValueError:
        # a JSON hiccup is not worth losing the turn: ask once more for plain
        # markdown (this catches the malformed-quote case, not a dead network)
        text = llm.chat(messages + [{"role": "user",
                                     "content": "只输出 Markdown 正文，不要 JSON。"}],
                        temperature=0.35, max_tokens=max_tokens)
        parsed, needs, query = False, False, ""
    return {
        "answer": text.strip(),
        "needs_new_search": needs,
        "search_query": query,
        "json_parsed": parsed,
        "duration_s": round(time.monotonic() - t0, 1),
    }
