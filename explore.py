"""Field exploration: turn "I want to understand X" into a tree of directions.

The ask mode answers a question. This module is for the step before that: the
user has a *field*, not a question, and does not yet know which part of it is
worth reading. It proposes a handful of directions, each one grounded in a few
landmark papers so the choice is made on evidence instead of on a
plausible-sounding label, and lets the user drill into one -- or ask for the
review right away.

Nothing here retrieves the final corpus. `survey_queries()` hands the chosen
directions back to the ordinary pipeline, so the review that comes out is
produced and verified by exactly the same chain as any other answer.
"""

import re

import llm
import search

# Preview papers come from the two sources that can rank by citation count and
# answer fast; a preview is a UI affordance, not the retrieval for the report.
PREVIEW_SOURCES = ("semantic_scholar", "openalex")
PREVIEW_PER_DIRECTION = int(3)
PREVIEW_POOL = int(8)   # relevance hits to choose the "代表作" from
MAX_DIRECTIONS = 6
MAX_SUB_DIRECTIONS = 5
MIN_DIRECTIONS = 3
SURVEY_QUERY_CAP = int(10)

_DIRECTION_SPEC = """每条方向是一个对象：
{"name": "中文方向名", "en": "English term", "why": "一句话说明它的角色",
 "queries": ["英文检索词", "英文检索词"]}

硬性规则：
1. name 用中文，不超过 14 字，必须是同行会用的说法，不能写成一整句话。
2. en 是该方向的标准英文术语（2-6 个词），报告的小标题会用它。
3. why 一句话，不超过 40 字：这个方向在回答用户的问题时承担什么角色。
4. queries 是 2-3 条英文检索词，直接能喂给 OpenAlex / Semantic Scholar；
   要带上领域限定词，不要出现 "review"、"research" 这类只有泛义的词。
5. 方向之间互不重叠、合起来覆盖主要脉络：核心机制、主流方法、关键应用、
   争议与不确定性、前沿进展。宁可少而实，不要凑数。
6. 不要重复【已经给过的方向】里的任何一条。
7. 只输出 JSON，不要解释。"""

SCOPE_PROMPT = """你是 ScholarMate 的科研导航助手。用户给你的是一个**领域或大问题**，\
还不是一个能直接回答的细问题。你要把这片领域切成可以分别调研的方向交给用户挑选，\
而不是直接作答。

输出：{{"directions": [ ... ]}}，{lo}-{hi} 条。

{spec}

补充：用户还没有往下走过，可能给出【探索路径】和【已经给过的方向】（用于往深处走时）。\
即使为空也要遵守第 6 条。"""

DEEPEN_PROMPT = """你是 ScholarMate 的科研导航助手。用户已经选中了一个方向，要把它往下拆。\
你给出的必须是**更具体、更可检索**的下一层方向，而不是换种说法重复父方向。

输出：{{"directions": [ ... ]}}，{lo}-{hi} 条。

{spec}

补充要求：
1. 每一条都必须是父方向的真子集——它的一个子问题、一种方法路线、一类数据或模型、\
一个争议点；换个说法的同义改写不算。
2. 越往下越具体：第二层就应该落到具体的技术名词、数据类型或实验体系上。
3. 父方向自身的检索词不要原样再给一遍。"""


def _clip(text, limit):
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _clean_query(q):
    q = " ".join(str(q or "").split())
    return q[:160]


def normalize(raw, parent=None, depth=0):
    """Validate the model's direction list into the shape the rest of the code
    (and the UI) expects. Junk entries are dropped rather than passed on."""
    if isinstance(raw, dict):
        raw = raw.get("directions") or raw.get("items") or []
    if not isinstance(raw, list):
        return []
    out, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()[:40]
        en = str(item.get("en") or item.get("english") or "").strip()[:80]
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        queries = [_clean_query(q) for q in (item.get("queries") or []) if _clean_query(q)]
        if not queries and en:
            queries = [en]
        if not queries:
            continue
        out.append({
            "id": f"{len(out) + 1}",
            "name": name,
            "en": en,
            "why": _clip(item.get("why") or item.get("role"), 80),
            "queries": queries[:3],
            "parent": (parent or {}).get("name", ""),
            "depth": depth,
            "refs": [],
        })
    return out


def _path_text(path):
    return " → ".join(path) if path else "（还没有往下走）"


def _seen_text(seen):
    return "、".join(seen[-24:]) if seen else "（无）"


def _ask(prompt, payload, max_tokens=1600):
    return llm.chat_json(
        [{"role": "system", "content": prompt},
         {"role": "user", "content": payload}],
        temperature=0.4, max_tokens=max_tokens,
    )


def scope(topic, path=None, seen=None, limit=MAX_DIRECTIONS):
    """Split a field into directions the user can choose between."""
    prompt = SCOPE_PROMPT.format(lo=MIN_DIRECTIONS, hi=limit, spec=_DIRECTION_SPEC)
    payload = (f"【领域/问题】{topic}\n\n"
               f"【探索路径】{_path_text(path)}\n\n"
               f"【已经给过的方向】{_seen_text(seen)}\n\n"
               f"请给出 {MIN_DIRECTIONS}-{limit} 个方向，只输出 JSON。")
    raw = _ask(prompt, payload)
    return normalize(raw)


def deepen(direction, path=None, seen=None, limit=MAX_SUB_DIRECTIONS):
    """Split one chosen direction into the next, more specific, layer."""
    direction = direction or {}
    parent = f"{direction.get('name', '')}（{direction.get('en', '')}）"
    prompt = DEEPEN_PROMPT.format(lo=MIN_DIRECTIONS, hi=limit, spec=_DIRECTION_SPEC)
    payload = (f"【父方向】{parent}\n"
               f"父方向的一句话说明：{direction.get('why', '')}\n"
               f"父方向的检索词：{'；'.join(direction.get('queries') or [])}\n\n"
               f"【探索路径】{_path_text(path)}\n\n"
               f"【已经给过的方向】{_seen_text(seen)}\n\n"
               f"请给出 {MIN_DIRECTIONS}-{limit} 个更具体的下一层方向，只输出 JSON。")
    raw = _ask(prompt, payload)
    depth = int(direction.get("depth") or 0) + 1
    return normalize(raw, parent=direction, depth=depth)


_STOP = {"the", "and", "for", "with", "from", "using", "based", "via", "into", "its",
         "are", "was", "were", "this", "that", "these", "large", "model", "models",
         "language", "new", "novel", "study", "review", "analysis", "approach",
         "method", "methods", "towards", "toward", "can", "not", "you", "your"}


def _tokens(text):
    return {w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
            if w not in _STOP}


def _ref(paper):
    return {
        "title": paper.get("title") or "",
        "year": paper.get("year"),
        "citations": paper.get("citations"),
        "venue": paper.get("venue") or "",
        "url": paper.get("url") or "",
        "source": paper.get("source") or "",
    }


def preview(directions, per=PREVIEW_PER_DIRECTION, sources=None,
            pool=PREVIEW_POOL):
    """Attach representative papers to each direction, in one parallel batch.

    One relevance query per direction (the direction's first), then the most
    cited of those matches. Citation-*sorted* retrieval is deliberately not
    used here: on a loose match it happily returns a 1969 statistics paper,
    and a preview the user can see is wrong costs more trust than a preview
    that is merely unglamorous. A failure leaves every direction without refs
    -- the directions themselves are still perfectly usable.
    """
    pairs = []
    for d in directions or []:
        queries = d.get("queries") or []
        if queries:
            pairs.append((d, queries[0]))
    if not pairs:
        return directions or []
    try:
        details = search.search_many_detailed(
            [q for _, q in pairs], limit=pool,
            sources=list(sources or PREVIEW_SOURCES),
        )
    except Exception as e:                      # preview is best effort
        print(f"      [explore] preview failed: {e}")
        return directions or []
    for (d, _), det in zip(pairs, details):
        ranked = sorted(det.get("papers") or [],
                        key=lambda p: (p.get("citations") or 0), reverse=True)
        # a relevance query still drags in the occasional unrelated classic
        # (a systematic-review tool answering "knowledge graph completion");
        # keep only titles that share a content word with the direction.
        wanted = _tokens((d.get("en") or "") + " " + " ".join(d.get("queries") or []))
        refs, seen = [], set()
        for p in ranked:
            title = (p.get("title") or "").strip()
            if not title or title.lower() in seen:
                continue
            if wanted and not (_tokens(title) & wanted):
                continue
            seen.add(title.lower())
            refs.append(_ref(p))
            if len(refs) >= per:
                break
        if not refs:                            # filter too aggressive: show the top hits
            for p in ranked[:per]:
                if (p.get("title") or "").strip():
                    refs.append(_ref(p))
        d["refs"] = refs
    return directions or []


def survey_queries(directions, cap=SURVEY_QUERY_CAP):
    """Merge the chosen directions' queries in round-robin order.

    Round-robin instead of concatenation: with a cap in place, taking them
    direction by direction would spend the whole budget on the first one and
    the report would silently drop the rest of the outline.
    """
    columns = [[_clean_query(q) for q in (d.get("queries") or []) if _clean_query(q)]
               for d in (directions or [])]
    out, seen, i = [], set(), 0
    while len(out) < cap and any(i < len(c) for c in columns):
        for col in columns:
            if len(out) >= cap:
                break
            if i < len(col):
                key = col[i].lower()
                if key not in seen:
                    seen.add(key)
                    out.append(col[i])
        i += 1
    for d in (directions or []):                # a direction with no query at all
        if len(out) >= cap:
            break
        fallback = _clean_query(d.get("en") or d.get("name") or "")
        if fallback and fallback.lower() not in seen:
            seen.add(fallback.lower())
            out.append(fallback)
    return out


def outline(directions):
    """The report's section list, used in the writing prompt and the UI."""
    lines = []
    for i, d in enumerate(directions or [], 1):
        head = f"{i}. {d.get('name', '')}"
        if d.get("en"):
            head += f"（{d['en']}）"
        if d.get("why"):
            head += f" —— {d['why']}"
        lines.append(head)
    return "\n".join(lines)


def names(directions):
    return "、".join(d.get("name", "") for d in (directions or []))
