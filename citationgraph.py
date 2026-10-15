"""文献关系图谱：把一篇论文放回它所在的引用网络里。

一次检索交给你二十篇「相关」论文，但它们之间的关系是平的：看不出哪几篇是这
个领域的基石、哪几篇是后来长出来的。把这个网络画出来，这些问题就能一眼回答：

  * 圆圈大小 = 被引量（这个领域里有多少人靠它）
  * 圆圈颜色 = 发表年份（一条从旧到新的色带，蓝=早，红=新）
  * 连边粗细 = 两篇论文共同引用了多少篇文献（书目耦合），也就是「有多像」
  * 高亮的一圈 = 核心程度（见下）

三种角色，对应研究者捋脉络时的三个问题：

  seed        你选的起点
  prior       它引用的文献 —— 这个领域站在谁肩上
  derivative  引用它的文献 —— 后来的人接着往哪走

数据只来自 OpenAlex：免费、无需 key，而且是唯一一个白送 ``referenced_works``
（每篇论文的参考文献表）的公开源。有了它，两篇论文「像不像」不用调模型，直接
数共同引用即可，可复现、可解释。

「核心」不用被引量，而用「这个领域有多少人和它一起被引用」：

  * 先行文献(prior)：种子论文的引用者里，有多少篇同时也引用了它（共被引次数）
  * 后续文献(derivative)：它引用了本领域（种子的参考文献表）里多少篇奠基工作

两边都是数共同引用，所以一篇被引量不高、但把两个方向缝在一起的论文会被顶起来
—— 这正是被引量排名看不见的东西。

另外，光按被引量排序会把「用过这个工具但并不在这个领域」的论文顶到最前面：以
NumPy 为种子，被引 4 万次的 AlphaFold2 和引力波论文都引用它，但它们与 NumPy
的参考文献几乎不重叠。所以后续文献先按「引用了多少篇本领域奠基工作」筛一遍，
筛掉的记为 off_field，在图谱上方如实报给读者，而不是悄悄混进去。

代价：一个约 80 节点的图需要 5-8 次 HTTP（种子 1 次、参考文献按 50 一批、引用
它的文献 1 页、补水合 1-2 批），实测 4-10 秒。相似度是 O(n^2) 的集合求交，全在
本地内存里算，80 个节点不到一秒。
"""

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import search


class GraphError(RuntimeError):
    """Worth telling the reader about -- as opposed to "this paper is not here",
    which is a normal answer and comes back as None."""

WORKS_URL = "https://api.openalex.org/works"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
# without ?fields= the batch endpoint answers with paperId and title only, so a
# request missing this silently comes back with no reference lists at all
S2_BATCH_FIELDS = "title,year,externalIds,references.externalIds"

MAX_PRIOR = 40             # how many of the seed's references to keep
MAX_DERIV = 40             # how many citing works to keep
MIN_COUPLING = 2           # two papers must share this many references to be linked
MAX_EDGES_PER_NODE = 4     # more than this and the picture turns into spaghetti
MIN_FIELD_HITS = 1         # a citing work must build on >=1 of the field's priors
MIN_DERIV_KEPT = 5         # ...unless that leaves almost nothing, then keep citations
BATCH = 50                 # OpenAlex OR-filter page size
LOOKUP_TIMEOUT = 30.0
MAX_REFS_KEPT = 400        # a 2000-reference review would cost set math for nothing

POOL_MAX = 40              # how many of this round's papers we try to place
POOL_MIN_SHARED = 2        # a reference must be shared by this many pool papers
POOL_MAX_FOUNDATION = 24   # ...and we draw at most this many, most shared first
POOL_LINKS_PER_PAPER = 5   # dashed lines per pool paper: more is spaghetti
POOL_ENRICH_BUDGET = 6.0   # how long the gap-filling pass may keep starting chunks
GAP_TIMEOUT = 8.0          # ...and how long one of its requests may take

# every field the graph needs, in one place: the same select string is used for
# the seed, the references and the citing works, so every node arrives carrying
# the reference list the coupling pass needs (one round trip per 50 nodes)
SELECT = ("id,doi,title,publication_year,cited_by_count,referenced_works,"
          "primary_location,authorships,type")


def _get(url, params, timeout=None):
    """GET with a couple of retries, for the sake of one TLS handshake.

    search._http_get_text deliberately does *not* retry connection-level
    failures (in a parallel query set that just doubles the wait for the
    slowest source), but this feature makes a handful of requests that a reader
    is actively waiting on, and losing one of them costs a whole half of the
    picture. Measured here: roughly one request in ten dies with
    SSL: UNEXPECTED_EOF_WHILE_READING, and succeeds on the next try.

    ``timeout`` is the caller's to tighten: the whole-round gap fill is a
    top-up rather than the main event, and must not hold the panel for a
    half-minute per request (see ``GAP_TIMEOUT``).
    """
    last = None
    for attempt in range(3):
        try:
            search.openalex_auth(params)
            return search.http_get_json(url, params, retries=1,
                                        timeout=timeout or LOOKUP_TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                raise           # a real answer: no such work / bad filter
            last = e
        except Exception as e:
            last = e
        if attempt < 2:
            time.sleep(0.7 * (attempt + 1))
    raise GraphError(f"OpenAlex 请求失败（网络不稳定）：{last}")


def _id_of(work_or_url):
    return str(work_or_url or "").rsplit("/", 1)[-1].strip()


def _brief(work):
    """One work -> one node. `refs` stays here for the coupling pass and is
    stripped before the payload leaves this module."""
    loc = work.get("primary_location") or {}
    source = (loc.get("source") or {})
    authors = [
        ((a.get("author") or {}).get("display_name") or "")
        for a in (work.get("authorships") or [])
    ]
    refs = [_id_of(x) for x in (work.get("referenced_works") or [])]
    doi = (work.get("doi") or "").replace("https://doi.org/", "").replace("http://doi.org/", "")
    return {
        "id": _id_of(work.get("id")),
        "title": (work.get("title") or "").strip(),
        "year": work.get("publication_year"),
        "citations": int(work.get("cited_by_count") or 0),
        "doi": doi,
        "venue": source.get("display_name") or "",
        "authors": [a for a in authors if a][:2],
        "type": work.get("type") or "",
        "refs": refs[:MAX_REFS_KEPT],
        # a work whose reference list the API did not return (some records have
        # none) must not be mistaken for "shares nothing with this field"
        "refs_known": "referenced_works" in work,
    }


def _looks_like_doi(text):
    return bool(re.match(r"^(doi:)?10\.\d{4,9}/\S+$", text, re.I))


def _looks_like_oa_id(text):
    return bool(re.match(r"^(https?://openalex\.org/)?[Ww]\d{5,}$", text))


def _s2_doi(title):
    """Last resort for a title: let Semantic Scholar name the DOI.

    OpenAlex indexes DOIs far more reliably than titles, so a round trip through
    S2 turns "a half-remembered paper name" into something OpenAlex can hit.
    """
    params = {"query": title[:200], "limit": 3, "fields": "title,externalIds,year"}
    headers = None
    if search.S2_API_KEY:
        headers = {"x-api-key": search.S2_API_KEY}
    try:
        data = search.http_get_json(S2_SEARCH_URL, params, timeout=LOOKUP_TIMEOUT,
                                    retries=1, headers=headers)
    except Exception:
        return ""
    want = re.sub(r"[^a-z0-9]+", "", title.lower())
    for paper in (data.get("data") or []):
        got = re.sub(r"[^a-z0-9]+", "", (paper.get("title") or "").lower())
        if not got:
            continue
        if got == want or got.startswith(want[:40]) or want.startswith(got[:40]):
            return ((paper.get("externalIds") or {}).get("DOI") or "").strip()
    return ""


def _work_by_doi(doi):
    """DOI -> node. None when OpenAlex genuinely does not have that DOI (404) --
    a network failure raises instead, so the two never get confused."""
    try:
        return _brief(_get(WORKS_URL + "/doi:" + doi, {"select": SELECT}))
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):
            return None
        raise


INFO_SELECT = ("id,doi,title,publication_year,cited_by_count,primary_location,"
               "authorships,type,abstract_inverted_index,best_oa_location,open_access,"
               "is_retracted")


def _paper(work):
    """One work -> the record a reader gets after clicking its circle.

    Shaped like a search result (``title``/``authors``/``abstract``/``doi``/
    ``url``/``pdf_url``/...) so the page can reuse the same card and the same
    link row it already shows for a round's papers.
    """
    loc = work.get("primary_location") or {}
    oa = work.get("best_oa_location") or {}
    authors = [((a.get("author") or {}).get("display_name") or "")
               for a in (work.get("authorships") or [])]
    doi = (work.get("doi") or "").replace("https://doi.org/", "").replace("http://doi.org/", "")
    oa_id = _id_of(work.get("id"))
    return {
        "id": oa_id,
        "title": (work.get("title") or "").strip(),
        "year": work.get("publication_year"),
        "citations": int(work.get("cited_by_count") or 0),
        "doi": doi,
        "venue": ((loc.get("source") or {}).get("display_name") or ""),
        "authors": [a for a in authors if a],
        "type": work.get("type") or "",
        "abstract": search.reconstruct_abstract(work.get("abstract_inverted_index")),
        "url": ("https://doi.org/" + doi) if doi else ("https://openalex.org/" + oa_id),
        "pdf_url": oa.get("pdf_url") or "",
        "oa_status": ((work.get("open_access") or {}).get("oa_status") or ""),
        "retracted": bool(work.get("is_retracted")),
        "openalex_url": "https://openalex.org/" + oa_id,
    }


def paper_info(key):
    """A graph node (OpenAlex id / DOI / title) -> the paper itself, or None.

    The graph payload is deliberately lean -- abstracts for forty foundations
    would triple it for a view that only needs circles -- so a click asks for
    the one record behind the circle it landed on. Keys are the same three the
    graph accepts as a seed, so "what is this circle" and "rebuild from this
    circle" always agree on which paper they mean.
    """
    text = (key or "").strip()[:400]
    if not text:
        return None
    try:
        if _looks_like_oa_id(text):
            work = _get(WORKS_URL + "/" + _id_of(text), {"select": INFO_SELECT})
        elif _looks_like_doi(text):
            work = _get(WORKS_URL + "/doi:" + re.sub(r"^doi:", "", text, flags=re.I),
                        {"select": INFO_SELECT})
        else:
            # a title or a topic: resolve() already owns the S2 fallback that
            # turns a half-remembered title into something OpenAlex can hit
            node = resolve(text)
            if not node:
                return None
            work = _get(WORKS_URL + "/" + node["id"], {"select": INFO_SELECT})
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):
            return None
        raise
    return _paper(work)


def resolve(seed):
    """A paper (DOI / OpenAlex ID / title / topic) -> its node, or None.

    Tried in order of precision: an ID or DOI is exact, a title is a guess, and
    free text is allowed on purpose -- "draw me the map of this field" usually
    starts from the field's most relevant paper rather than from a citation the
    reader happens to have in hand.
    """
    text = (seed or "").strip()
    if not text:
        return None
    if _looks_like_oa_id(text):
        try:
            return _brief(_get(WORKS_URL + "/" + _id_of(text), {"select": SELECT}))
        except urllib.error.HTTPError:
            return None
    if _looks_like_doi(text):
        doi = re.sub(r"^doi:", "", text, flags=re.I)
        node = _work_by_doi(doi)
        if node:
            return node
        doi2 = _s2_doi(text)
        return _work_by_doi(doi2) if doi2 else None
    for params in ({"filter": "title.search:" + re.sub(r"[,:]", " ", text[:300]),
                    "per-page": 1, "select": SELECT},
                   {"search": text[:300], "per-page": 1, "select": SELECT}):
        try:
            data = _get(WORKS_URL, params)
        except urllib.error.HTTPError:
            continue
        for work in (data.get("results") or []):
            if (work.get("title") or "").strip():
                return _brief(work)
    doi = _s2_doi(text)
    return _work_by_doi(doi) if doi else None


def _hydrate(ids, workers=4):
    """Several OpenAlex IDs -> their nodes, in batches of 50 and in parallel.

    Silent about missing IDs: OpenAlex drops deleted records from an OR filter
    instead of failing, so a partial answer is the normal case here. Returns
    (nodes, failed_batches) so the caller can say out loud when the graph came
    out incomplete rather than presenting half a picture as a whole one.
    """
    ids = [i for i in dict.fromkeys(ids) if i]
    if not ids:
        return [], 0
    chunks = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]

    def one(chunk):
        try:
            data = _get(WORKS_URL, {"filter": "openalex_id:" + "|".join(chunk),
                                    "per-page": BATCH, "select": SELECT})
            return data.get("results") or [], 0
        except Exception:
            return [], 1

    raw, failed = [], 0
    if len(chunks) == 1:
        raw, failed = one(chunks[0])
    else:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(chunks)))) as pool:
            for part, bad in pool.map(one, chunks):
                raw += part
                failed += bad
    return [_brief(w) for w in raw if (w.get("title") or "").strip()], failed


def _citing(wid, limit):
    """Works that cite `wid`, most cited first -> (nodes, failed).

    Most cited first is only the *candidate* order: which of them belong to this
    field is decided later, by how much of the field's foundation they cite.
    """
    if not wid:
        return [], 0
    try:
        data = _get(WORKS_URL, {"filter": "cites:" + wid,
                                "sort": "cited_by_count:desc",
                                "per-page": min(50, max(1, limit)), "select": SELECT})
    except Exception:
        return [], 1
    return [_brief(w) for w in (data.get("results") or [])
            if (w.get("title") or "").strip()][:limit], 0


def _coupling(nodes, min_coupling, max_per_node):
    """Node-dict flavoured _couple(): the similarity pass for build()."""
    return _couple({n["id"]: set(n.get("refs") or []) for n in nodes},
                   min_coupling, max_per_node)


def build(seed, max_prior=MAX_PRIOR, max_deriv=MAX_DERIV,
          min_coupling=MIN_COUPLING, max_edges_per_node=MAX_EDGES_PER_NODE):
    """The whole picture: {seed, nodes, edges, core, year_range, stats}.

    Raises LookupError when the seed cannot be identified at all -- the only
    failure the UI has to explain to the reader.
    """
    t0 = time.time()
    seed_node = resolve(seed)
    if not seed_node:
        raise LookupError("找不到这篇论文：换成 DOI、OpenAlex ID 或更完整的标题再试")
    seed_refs = seed_node.pop("refs", [])
    seed_node.pop("refs_known", None)
    wid = seed_node["id"]
    seed_node["role"] = "seed"

    # backward and forward at the same time: neither depends on the other
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_prior = pool.submit(_hydrate, seed_refs[:200])
        f_deriv = pool.submit(_citing, wid, max_deriv)
        prior, prior_failed = f_prior.result()
        deriv, deriv_failed = f_deriv.result()
    warnings = []
    if seed_refs and prior_failed:
        warnings.append(f"参考文献表有 {prior_failed} 批没取到（网络抖动），先行文献可能不全")
    if deriv_failed:
        warnings.append("没能取到引用该论文的文献（网络抖动），图中只有先行文献")

    # Rank the seed's references by how many of the seed's own citers lean on
    # them too: that is the field's real foundation, and it beats raw citation
    # counts, which just surface the oldest blockbusters.
    deriv_refsets = [set(w.get("refs") or []) for w in deriv]
    for work in prior:
        work["support"] = sum(1 for rs in deriv_refsets if work["id"] in rs)
    prior.sort(key=lambda w: (-w["support"], -w["citations"], w["title"]))
    dropped_prior = max(0, len(prior) - max_prior)
    prior = prior[:max_prior]
    for work in prior:
        work["role"] = "prior"

    # Keep the citing works that actually build on this field: a paper that
    # merely uses the same tool shares none of the field's foundation, and
    # letting it in (it is usually a giant cross-domain paper) would bury the
    # papers the reader is looking for.
    prior_ids = {w["id"] for w in prior}
    have = set(prior_ids) | {wid}
    seen = set()
    on_field, off_field = [], 0
    for work in deriv:
        work["role"] = "derivative"
        work["field_hits"] = len(set(work.get("refs") or []) & prior_ids)
        if work["id"] in have or work["id"] in seen:
            continue
        seen.add(work["id"])
        if work["field_hits"] >= MIN_FIELD_HITS:
            on_field.append(work)
        else:
            off_field += 1
    on_field.sort(key=lambda w: (-w["field_hits"], -w["citations"], w["title"]))
    relaxed = False
    if len(on_field) < MIN_DERIV_KEPT:
        # a young field (or a tool paper) can have almost no citers that overlap
        # its bibliography; showing the top-cited ones beats showing nothing
        relaxed = True
        extra = [w for w in deriv if w["id"] not in {x["id"] for x in on_field}
                 and w["id"] not in have]
        extra.sort(key=lambda w: -w["citations"])
        on_field += extra[:max_deriv - len(on_field)]
    deriv = on_field[:max_deriv]

    # citing works arrive with their reference lists (the citing query selects
    # them), but a record may have none; fetch the ones the API left out so the
    # similarity pass does not silently read them as "shares nothing"
    unknown = [w["id"] for w in deriv if not w.get("refs_known")]
    if unknown:
        refilled, _ = _hydrate(unknown)
        for work in refilled:
            for node in deriv:
                if node["id"] == work["id"] and work.get("refs"):
                    node["refs"] = work["refs"]

    nodes = [seed_node] + prior + deriv
    coupled = _coupling(nodes, min_coupling, max_edges_per_node)

    edges = []
    for work in prior:
        edges.append({"source": wid, "target": work["id"], "kind": "cites",
                      "weight": 3.0, "dir": "back"})
    for work in deriv:
        edges.append({"source": work["id"], "target": wid, "kind": "cites",
                      "weight": 3.0, "dir": "fwd"})
    for shared, a, b in coupled:
        edges.append({"source": a, "target": b, "kind": "coupled",
                      "weight": float(min(shared, 10)), "dir": ""})

    # similarity to the seed: how much of its bibliography this paper shares
    seed_set = set(seed_refs)
    for node in nodes:
        node["sim"] = len(set(node.get("refs") or []) & seed_set)
    top_sim = max((n["sim"] for n in nodes), default=0) or 1

    # core: role-aware "how many people in this field cite it alongside the
    # others" -- priors by co-citation with the citing set, derivatives by how
    # much of the field's foundation they build on
    top_support = max([n.get("support", 0) for n in prior], default=0) or 1
    top_hits = max([n.get("field_hits", 0) for n in deriv], default=0) or 1
    for node in nodes:
        node.pop("refs", None)
        node.pop("refs_known", None)
        node["sim"] = round(node["sim"] / top_sim, 3)
        if node["role"] == "prior":
            node["core"] = round(node.get("support", 0) / top_support, 3)
        elif node["role"] == "derivative":
            node["core"] = round(node.get("field_hits", 0) / top_hits, 3)
        else:
            node["core"] = 1.0

    years = [n["year"] for n in nodes if n.get("year")]
    core = sorted([n for n in nodes if n["role"] != "seed"],
                  key=lambda n: (-n["core"], -n["citations"]))[:12]
    return {
        "seed": seed_node,
        "nodes": nodes,
        "edges": edges,
        "core": core,
        "year_range": [min(years), max(years)] if years else [],
        "stats": {
            "nodes": len(nodes), "edges": len(edges),
            "prior": len(prior), "derivative": len(deriv),
            "coupled_edges": len(coupled),
            "truncated_prior": dropped_prior,
            "off_field": off_field,
            "relaxed": relaxed,
            "warnings": warnings,
            "seconds": round(time.time() - t0, 1),
        },
    }


def _normalize_doi(text):
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", str(text or "").strip(),
                 flags=re.I)
    return doi if _looks_like_doi(doi) else ""


def _by_dois(dois, workers=4):
    """Several DOIs -> {doi_lower: node}, one OR-filter request per 50.

    This is what keeps a twenty-paper graph as cheap as a two-paper one. It also
    repairs itself: a DOI the batch did not return (odd characters, a record
    OpenAlex has dropped) is asked for on its own, in parallel and bounded, so a
    single bad entry cannot leave a hole in the picture.
    """
    dois = [d for d in dict.fromkeys(dois) if d]
    if not dois:
        return {}
    found = {}
    chunks = [dois[i:i + BATCH] for i in range(0, len(dois), BATCH)]

    def one(chunk):
        try:
            data = _get(WORKS_URL, {"filter": "doi:" + "|".join(chunk),
                                    "per-page": min(200, max(BATCH, len(chunk))),
                                    "select": SELECT})
        except Exception:
            return []
        return data.get("results") or []

    raw = []
    if len(chunks) == 1:
        raw = one(chunks[0])
    else:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(chunks)))) as pool:
            for part in pool.map(one, chunks):
                raw += part
    for work in raw:
        node = _brief(work)
        if node["doi"]:
            found[node["doi"].lower()] = node

    missing = [d for d in dois if d.lower() not in found]
    if missing and len(missing) <= BATCH:
        def solo(doi):
            try:
                node = _work_by_doi(doi)
            except Exception:
                return None
            return (doi.lower(), node) if node else None

        with ThreadPoolExecutor(max_workers=max(1, min(6, len(missing)))) as pool:
            for got in pool.map(solo, missing):
                if got:
                    found[got[0]] = got[1]
    return found


def _couple(refmap, min_coupling, max_per_node):
    """|refs(a) & refs(b)| for every pair, in memory.

    O(n^2) set intersections, all in memory -- at the node budgets this module
    works with that is a few milliseconds, so a co-citation index or a second
    round of API calls would be wasted effort.
    """
    ids = [i for i in refmap if refmap[i]]
    pairs = []
    for i, a in enumerate(ids):
        ra = refmap[a]
        for b in ids[i + 1:]:
            rb = refmap[b]
            shared = len(ra & rb)
            if shared >= min_coupling:
                pairs.append((shared, a, b))
    pairs.sort(key=lambda p: -p[0])
    degree = {i: 0 for i in ids}
    kept = []
    for shared, a, b in pairs:
        if degree[a] >= max_per_node or degree[b] >= max_per_node:
            continue
        degree[a] += 1
        degree[b] += 1
        kept.append((shared, a, b))
    return kept


def _doi_by_oa_id(ids, workers=4, page=100, timeout=None, deadline=None):
    """OpenAlex reference ids -> ({id: doi}, failed_chunks).

    Needed to put the two reference sources in one key space: OpenAlex hands out
    reference lists keyed by *its own* ids, Semantic Scholar by DOI/CorpusId, and
    two reference lists can only be intersected if they are written in the same
    alphabet.

    Chunks are 100 ids, not the 200 that ``per-page`` would allow: the OR filter
    travels in the URL, and OpenAlex answers 400 once the query string gets too
    long (measured: 100 ids fine, 150 refused). Failures are counted and reported
    rather than swallowed -- a silent empty answer here would just look like
    "these papers share nothing".

    ``deadline`` is for the whole-round gap fill, which is a top-up on top of
    another source: chunks that would only *start* after it are counted as
    failures instead of being quietly dropped, and ``timeout`` keeps a single
    one of them from running long enough to matter.
    """
    ids = [i for i in dict.fromkeys(ids) if i]
    if not ids:
        return {}, 0
    chunks = [ids[i:i + page] for i in range(0, len(ids), page)]

    def one(chunk):
        if deadline is not None and time.time() > deadline:
            return [], 1
        try:
            data = _get(WORKS_URL, {"filter": "openalex_id:" + "|".join(chunk),
                                    "per-page": page, "select": "id,doi"},
                        timeout=timeout)
        except Exception:
            return [], 1
        return data.get("results") or [], 0

    raw, failed = [], 0
    if len(chunks) == 1:
        raw, failed = one(chunks[0])
    else:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(chunks)))) as pool:
            for part, bad in pool.map(one, chunks):
                raw += part
                failed += bad
    out = {}
    for work in raw:
        doi = (work.get("doi") or "").replace("https://doi.org/", "").replace("http://doi.org/", "")
        if doi:
            out[_id_of(work.get("id"))] = doi.lower()
    return out, failed


def _ref_key(ext):
    """One reference -> one comparable key.

    A DOI when the reference has one (that is also what lets the foundation
    layer be hydrated through OpenAlex), otherwise Semantic Scholar's own corpus
    id. The rule is deterministic, so the *same* reference reaches the same key
    from both papers that cite it -- which is all the coupling pass needs.
    """
    ext = ext or {}
    doi = (ext.get("DOI") or "").strip().lower()
    if doi:
        return "doi:" + doi
    if ext.get("CorpusId"):
        return "s2:" + str(ext["CorpusId"])
    return ""


def _s2_handles(papers, max_pool):
    """The round's papers -> [(index, s2 id, doi_or_empty)] for the S2 batch.

    A DOI is the best handle, but a search round is full of arXiv preprints that
    have none (measured: 16 of 40 papers on one round), and dropping them would
    quietly shrink "the whole round" to a third of it. S2 accepts ``ARXIV:`` and
    ``PMID:`` ids and answers with the paper's real DOI, so those papers are
    resolved in the same POST that fetches everyone's reference lists.
    """
    out = []
    for index, paper in enumerate(papers[:max_pool]):
        doi = _normalize_doi(paper.get("doi"))
        if doi:
            out.append((index, "DOI:" + doi, doi))
            continue
        arxiv = re.sub(r"v\d+$", "", str(paper.get("arxiv_id") or "").strip())
        if arxiv:
            out.append((index, "ARXIV:" + arxiv, ""))
    return out


def _s2_refs(s2_ids, batch=100, retries=2, budget=6.0, timeout=12.0):
    """Semantic Scholar ids -> {id_lower: {"doi": str, "refs": set(keys)}}.

    One batched POST for the whole round. Semantic Scholar answers for nearly
    every paper a search returns, with roughly 40 references each, where
    OpenAlex's own ``referenced_works`` covers only part of them (measured:
    18/20 works on one round, 13/46 on another) -- and a coupling graph without
    reference lists has no edges at all.

    Returns {} when S2 will not answer: it rate-limits readily, and waiting it
    out is a bad trade here because OpenAlex can fill exactly the same gaps. The
    whole call is therefore held to a wall-clock ``budget``.
    """
    ids = [i for i in dict.fromkeys(s2_ids) if i]
    if not ids:
        return {}
    headers = {"Content-Type": "application/json"}
    if search.S2_API_KEY:
        headers["x-api-key"] = search.S2_API_KEY
    out = {}
    deadline = time.time() + budget
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        body = json.dumps({"ids": chunk}).encode("utf-8")
        data, ok = None, False
        url = S2_BATCH_URL + "?fields=" + S2_BATCH_FIELDS
        for attempt in range(retries):
            if time.time() > deadline:
                break
            try:
                req = urllib.request.Request(url, data=body, headers=dict(headers))
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                ok = True
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503):
                    time.sleep(min(1.5 * (attempt + 1), 3.0))
                    continue
                break
            except Exception:
                time.sleep(1.0 + attempt)
        if not ok or not isinstance(data, list):
            return {}
        for s2_id, entry in zip(chunk, data):
            if not entry:
                continue
            keys = {_ref_key(ref.get("externalIds")) for ref in (entry.get("references") or [])}
            keys.discard("")
            doi = ((entry.get("externalIds") or {}).get("DOI") or "").strip().lower()
            if keys or doi:
                out[s2_id.lower()] = {"doi": doi, "refs": keys}
    return out


def build_pool(papers, max_pool=POOL_MAX, min_shared=POOL_MIN_SHARED,
               max_foundation=POOL_MAX_FOUNDATION,
               min_coupling=MIN_COUPLING, max_edges_per_node=MAX_EDGES_PER_NODE):
    """A whole round of results -> one graph.

    ``build()`` answers "what does this paper sit on, and what grew out of it"
    for one paper. In front of a list of twenty results the question is
    different: which of these are the same story, which one is the hinge, and
    what do they all stand on. So the nodes here are

      pool   this round's own papers (the ones you actually searched for)
      prior  the works cited by >= ``min_shared`` of them -- their shared
             foundation, i.e. the literature this batch is collectively built on

    and the edges are the citations visible between them plus the bibliography
    coupling between the pool papers (again: "how alike", counted, not guessed).

    Two sources, one job each, on purpose:

      Semantic Scholar  reference lists, in one batched POST for the whole round.
      OpenAlex          reference lists too (its ``referenced_works`` covers the
                        papers S2 misses, and vice versa -- measured: 18/20 vs
                        15/26 on the same round), plus all the metadata, hence
                        every circle: year, venue and the citation count that
                        sizes the node. Circle size must not compare two
                        different sources' tallies, so all nodes get theirs from
                        OpenAlex even when their edges came from S2.

    The two reference sources are then merged into one DOI key space: OpenAlex
    speaks in its own work ids, S2 in DOI/CorpusId, and two reference lists can
    only be intersected once they are written in the same alphabet.

    Citing works ("who cites these twenty papers") are deliberately not fetched:
    that is twenty more queries, and the answering half of the picture is what
    this view is for.

    Core is again "position in *this* picture", not citation count, and again
    role-aware, so a 1.0 means "the strongest of its own kind here":

      pool        how many of the round's other papers it is drawn connected to
      prior       how many of the round's papers cite it (a shared foundation
                  cited by half the batch is the definition of load-bearing)
    """
    t0 = time.time()
    warnings = []

    # Which of the round's papers can be placed at all: a DOI is best, an arXiv
    # id is enough for S2 to name one. Papers with neither cannot be looked up
    # in any of the sources this module uses, and are counted rather than
    # silently dropped from "the whole round".
    candidates, no_id, seen = [], 0, set()
    for paper in (papers or []):
        paper = paper or {}
        doi = _normalize_doi(paper.get("doi"))
        arxiv = re.sub(r"v\d+$", "", str(paper.get("arxiv_id") or "").strip())
        key = (doi or arxiv).lower()
        if not key:
            no_id += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        candidates.append(paper)
    if not candidates:
        raise LookupError("这一批论文既没有 DOI 也没有 arXiv 号，画不出关系图：换个检索词，"
                          "或对单篇用卡片上的「🕸 关系图谱」")
    truncated = max(0, len(candidates) - max_pool)
    candidates = candidates[:max_pool]
    handles = _s2_handles(candidates, len(candidates))

    # The two lookups do not depend on each other -- S2 only needs the handles
    # the reader's browser already sent -- so they run together rather than in
    # line.
    payload_dois = [d for d in (_normalize_doi(p.get("doi")) for p in candidates) if d]
    with ThreadPoolExecutor(max_workers=2) as par:
        f_meta = par.submit(_by_dois, payload_dois)
        f_refs = par.submit(_s2_refs, [s2_id for _, s2_id, _ in handles])
        by_doi, s2 = f_meta.result(), f_refs.result()

    # Papers that had no DOI but did have an arXiv id: S2 just named their DOI,
    # which is one more batched OpenAlex call -- only made when such a paper is
    # actually in the round.
    recovered = []
    for _, s2_id, _ in handles:
        info = s2.get(s2_id.lower()) or {}
        if info.get("doi") and info["doi"] not in by_doi:
            recovered.append(info["doi"])
    if recovered:
        by_doi.update(_by_dois(recovered))

    pool, refmap = [], {}
    for index, s2_id, doi in handles:
        info = s2.get(s2_id.lower()) or {}
        node = by_doi.get((doi or info.get("doi") or "").lower())
        if not node:
            continue
        node["role"] = "pool"
        if not node["venue"]:
            node["venue"] = str(candidates[index].get("venue") or "")
        if not node["year"]:
            node["year"] = candidates[index].get("year")
        pool.append(node)
        if info.get("refs"):
            refmap[node["id"]] = set(info["refs"])
    from_s2 = len(refmap)
    dropped = len(handles) - len(pool)
    if dropped:
        warnings.append(f"{dropped} 篇在 OpenAlex 里没有匹配到（或已被删除），没有画进图里")
    if no_id:
        warnings.append(f"{no_id} 篇既没有 DOI 也没有 arXiv 号，无法定位")
    if truncated:
        warnings.append(f"本轮论文超过图上限，只画了前 {max_pool} 篇"
                        f"（另有 {truncated} 篇没画进来）")
    if len(pool) < 2:
        raise LookupError("匹配到的论文不足 2 篇，画不出关系图：换个检索词，"
                          "或对单篇用卡片上的「🕸 关系图谱」")

    # S2 carries most reference lists, but not all (measured: it answered for
    # 20/26 papers on one round and 14/31 on another). OpenAlex's own lists fill
    # exactly those gaps -- and only those, because OpenAlex writes them in its
    # own work ids, and turning those into DOIs costs a request per hundred
    # references. Converting *every* paper's refs that way was measured to add
    # nothing over converting just the gaps, and cost two extra seconds.
    gaps = [n for n in pool if n["id"] not in refmap]
    gap_ids = [r for n in gaps for r in (n.get("refs") or [])]
    # This pass is what stands between a rate-limited Semantic Scholar and a
    # graph with no relations in it at all, so it is not skipped just because S2
    # spent the clock on its own budget: it gets a ceiling of its own (no new
    # chunk is started after it) instead of a yes/no on time already spent.
    # Measured on the 20-paper round with S2 held quiet: skipping it came back
    # with "20 nodes / 0 relations" in 6s, running it with a ceiling comes back
    # with "20 + 19 nodes / 38 relations" in 10-16s. A picture with no relations
    # in it is not worth the 6s it saved.
    oa_doi, oa_failed = _doi_by_oa_id(gap_ids, timeout=GAP_TIMEOUT,
                                      deadline=time.time() + POOL_ENRICH_BUDGET)
    if oa_failed:
        warnings.append(f"OpenAlex 参考文献里有 {oa_failed} 批没能换成 DOI（网络抖动，"
                        f"或没排进 {POOL_ENRICH_BUDGET:.0f}s 预算），这些论文的连线可能偏少")
    from_oa = 0
    for node in gaps:
        keys = set()
        for ref in (node.get("refs") or []):
            doi = oa_doi.get(ref)
            keys.add("doi:" + doi if doi else "oa:" + ref)
        refmap[node["id"]] = keys
        if keys:
            from_oa += 1
    refs_coverage = sum(1 for v in refmap.values() if v)
    if not refs_coverage:
        warnings.append("这一批论文的参考文献表两个源都没有收录，没有可算的关系："
                        "图里只剩「年份 + 被引」两个维度")
    elif refs_coverage < len(pool) * 0.6:
        warnings.append(f"只有 {refs_coverage}/{len(pool)} 篇取到了参考文献表，"
                        f"其余在图里没有连线（数据里就没有，不是没查）")

    # shared foundation: count, per reference, how many of the round's papers
    # lean on it. The count is also what decides which ones get drawn.
    pool_keys = {"doi:" + (n["doi"] or "").lower() for n in pool if n["doi"]}
    counts = {}
    for keys in refmap.values():
        for key in keys:
            if key not in pool_keys:
                counts[key] = counts.get(key, 0) + 1
    shared = sorted(((c, k) for k, c in counts.items() if c >= min_shared),
                    key=lambda pair: (-pair[0], pair[1]))
    # Foundation nodes are hydrated through OpenAlex (metadata has to come from
    # one source or the circle sizes would compare two different tallies), so
    # only DOI-keyed references can be drawn; the rest still count towards
    # coupling but cannot be shown as a node.
    want = [k[4:] for _, k in shared if k.startswith("doi:")][:max_foundation]
    foundation = list(_by_dois(want).values())
    key_of = lambda node: "doi:" + (node["doi"] or "").lower()
    for node in foundation:
        node["role"] = "prior"
        node["support"] = counts.get(key_of(node), 0)
    foundation.sort(key=lambda n: (-n["support"], -n["citations"], n["title"]))
    fund_by_key = {key_of(n): n for n in foundation}
    undrawable = sum(1 for _, k in shared if not k.startswith("doi:"))
    if undrawable:
        warnings.append(f"{undrawable} 篇被共同引用的文献在 OpenAlex 里查不到 DOI，没有画进图里"
                        "（它们仍在计算「有多像」时被算进去）")

    # citations between the round's papers: plain membership, no request
    citing = []
    for node in pool:
        keys = refmap.get(node["id"]) or set()
        for other in pool:
            if other is not node and ("doi:" + (other["doi"] or "").lower()) in keys:
                citing.append((node["id"], other["id"]))
    coupled = _couple(refmap, min_coupling, max_edges_per_node)

    edges = []
    for node in pool:
        keys = refmap.get(node["id"]) or set()
        linked = sorted((fund_by_key[k] for k in keys if k in fund_by_key),
                        key=lambda n: (-n["support"], -n["citations"]))
        for target in linked[:POOL_LINKS_PER_PAPER]:
            edges.append({"source": node["id"], "target": target["id"], "kind": "cites",
                          "weight": 3.0, "dir": "back"})
    for a, b in citing:
        edges.append({"source": a, "target": b, "kind": "cites",
                      "weight": 3.0, "dir": "back"})
    for shared_n, a, b in coupled:
        edges.append({"source": a, "target": b, "kind": "coupled",
                      "weight": float(min(shared_n, 10)), "dir": ""})

    # core: how connected it is inside *this* picture. For a pool paper that is
    # the number of drawn neighbours (peers it couples with, or it cites); for a
    # foundation work it is how many of the batch cite it. Both are normalised
    # within their own kind, exactly as build() does for prior/derivative.
    peers = {n["id"]: set() for n in pool}
    for _, a, b in coupled:
        peers[a].add(b)
        peers[b].add(a)
    for a, b in citing:
        peers[a].add(b)
        peers[b].add(a)
    best = {}
    for shared_n, a, b in coupled:
        best[a] = max(best.get(a, 0), shared_n)
        best[b] = max(best.get(b, 0), shared_n)
    top_peers = max((len(v) for v in peers.values()), default=0) or 1
    top_support = max((n["support"] for n in foundation), default=0) or 1
    top_best = max(best.values(), default=0) or 1

    nodes = pool + foundation
    for node in nodes:
        node.pop("refs", None)
        node.pop("refs_known", None)
        if node["role"] == "pool":
            node["core"] = round(len(peers[node["id"]]) / top_peers, 3)
            node["sim"] = round(best.get(node["id"], 0) / top_best, 3)
        else:
            node["core"] = round(node["support"] / top_support, 3)
            node["sim"] = None        # "how much of it the seed shares" means nothing here

    isolated = sum(1 for n in pool if not peers[n["id"]] and
                   not any(e["source"] == n["id"] or e["target"] == n["id"] for e in edges))
    if isolated:
        warnings.append(f"{isolated} 篇与本批其它论文查不到共同引用（常是新近论文，"
                        f"或参考文献表缺失），在图里是孤点")

    years = [n["year"] for n in nodes if n.get("year")]
    core = sorted(nodes, key=lambda n: (-n["core"], -n["citations"]))[:12]
    return {
        "kind": "pool",
        "seed": {"id": "pool", "title": f"本轮检索的 {len(pool)} 篇论文",
                 "year": None, "citations": 0, "doi": "", "venue": "",
                 "authors": [], "type": "", "role": "pool", "core": 1.0, "sim": None},
        "nodes": nodes,
        "edges": edges,
        "core": core,
        "year_range": [min(years), max(years)] if years else [],
        "stats": {
            "nodes": len(nodes), "edges": len(edges),
            "pool": len(pool), "foundation": len(foundation),
            "coupled_edges": len(coupled), "citing_edges": len(citing),
            "shared_refs": len(shared),
            "refs_s2": from_s2, "refs_openalex": from_oa,
            "refs_coverage": refs_coverage, "isolated_pool": isolated,
            "unresolved": dropped, "no_id": no_id, "truncated": truncated,
            "warnings": warnings,
            "seconds": round(time.time() - t0, 1),
        },
    }


def summary(payload):
    """One line for the run log / UI chip."""
    if not payload:
        return "图谱：无数据"
    s = payload.get("stats") or {}
    if payload.get("kind") == "pool":
        return (f"图谱：本轮 {s.get('pool', 0)} 篇 + 共同奠基 {s.get('foundation', 0)} 篇，"
                f"{s.get('nodes', 0)} 节点 / {s.get('edges', 0)} 条关系"
                f"（{s.get('seconds', 0)}s）")
    seed = payload.get("seed") or {}
    return (f"图谱：以《{(seed.get('title') or '')[:34]}》为中心，"
            f"{s.get('nodes', 0)} 篇文献 / {s.get('edges', 0)} 条关系"
            f"（先行 {s.get('prior', 0)} · 后续 {s.get('derivative', 0)}，"
            f"{s.get('seconds', 0)}s）")
