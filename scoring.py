"""Multi-dimensional evidence scoring with user-tunable weights (pure stdlib).

One "relevance" number cannot express what makes evidence usable for a specific
question: a mechanistic claim needs different support than a clinical one. Each
paper is therefore scored 0-100 on ten dimensions and combined with weights the
user controls:

    相关性 / 直接性 / 方法可靠性 / 重复验证 / 结构支持 / 数据完整性 /
    物种匹配 / 影响力 / 偏倚风险 / 矛盾证据

Two conventions keep the numbers unambiguous:
  * every dimension is 0-100, and for the two *risk* dimensions (偏倚风险、
    矛盾证据) 100 means "most risky";
  * the weighted total folds risk in as (100 - score), so the total is always
    "higher is better" regardless of which dimensions a user turns up.

Nine dimensions are judged by DeepSeek from the abstract; 影响力 is computed
locally from citation velocity and venue, so it needs no model call and cannot
drift between runs. Contradictions are additionally detected *across* the
result set in a second pass, because no single abstract can reveal that another
paper disagrees with it.
"""

import contextvars
import atomicio
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import llm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.environ.get("SCORING_SETTINGS") or os.path.join(
    BASE_DIR, "outputs", "scoring_weights.json")

# how many papers one LLM call scores, and how many papers we are willing to
# score in a single run (search mode can return 90+ papers; scoring all of them
# would dwarf the rest of the pipeline)
BATCH_SIZE = int(os.environ.get("SCORING_BATCH_SIZE", "4"))
MAX_PAPERS = int(os.environ.get("SCORING_MAX_PAPERS", "40"))
WORKERS = int(os.environ.get("SCORING_WORKERS", "4"))
ABSTRACT_CHARS = 700

# direction: +1 the higher the better, -1 the higher the riskier
DIMENSIONS = [
    {"key": "relevance", "label": "相关性", "direction": 1, "default": 5,
     "hint": "是否直接回答用户问题"},
    {"key": "directness", "label": "直接性", "direction": 1, "default": 4,
     "hint": "是否为直接实验证据（而非综述转述或纯推论）"},
    {"key": "method", "label": "方法可靠性", "direction": 1, "default": 5,
     "hint": "实验方法是否适合支持该结论"},
    {"key": "replication", "label": "重复验证", "direction": 1, "default": 4,
     "hint": "是否被多篇文献或独立团队验证"},
    {"key": "structural", "label": "结构支持", "direction": 1, "default": 1,
     "hint": "是否有结构生物学证据（晶体/冷冻电镜/结构预测）"},
    {"key": "data", "label": "数据完整性", "direction": 1, "default": 4,
     "hint": "是否有完整数据与统计检验"},
    {"key": "species", "label": "物种匹配", "direction": 1, "default": 3,
     "hint": "是否符合用户目标物种（未填目标物种时不计入）"},
    {"key": "impact", "label": "影响力", "direction": 1, "default": 2,
     "hint": "引用、期刊、作者等影响力（本地计算：引用速度 + 发表载体）"},
    {"key": "bias", "label": "偏倚风险", "direction": -1, "default": 4,
     "hint": "是否存在实验缺陷或利益冲突（分数越高风险越大）"},
    {"key": "contradiction", "label": "矛盾证据", "direction": -1, "default": 4,
     "hint": "是否存在相反结论（分数越高冲突越大）"},
]
DIM_KEYS = [d["key"] for d in DIMENSIONS]
BY_KEY = {d["key"]: d for d in DIMENSIONS}
# judged by the model; everything else is either computed locally (impact) or
# derived from the cross-paper pass (contradiction)
LLM_KEYS = [k for k in DIM_KEYS if k not in ("impact",)]

GRADES = [(80, "强", "good"), (65, "较强", "good"),
          (50, "中等", "warn"), (35, "偏弱", "bad"), (0, "弱", "bad")]

PRESETS = {
    "balanced": {"label": "均衡（默认）", "weights": {}},
    "mechanism": {
        "label": "机制 / 结构优先",
        "weights": {"structural": 5, "method": 5, "data": 4, "directness": 4,
                    "relevance": 4, "replication": 4, "species": 3,
                    "bias": 4, "contradiction": 3, "impact": 2},
    },
    "experimental": {
        "label": "实验验证优先",
        "weights": {"directness": 5, "method": 5, "data": 5, "replication": 5,
                    "bias": 5, "contradiction": 4, "relevance": 4, "species": 2,
                    "impact": 1, "structural": 1},
    },
    "clinical": {
        "label": "临床 / 转化",
        "weights": {"species": 5, "relevance": 5, "bias": 5, "directness": 4,
                    "method": 4, "replication": 4, "data": 4, "contradiction": 4,
                    "impact": 3, "structural": 0},
    },
    "screening": {
        "label": "快速初筛",
        "weights": {"relevance": 5, "impact": 3, "directness": 2, "method": 2,
                    "replication": 1, "data": 1, "species": 1, "structural": 0,
                    "bias": 1, "contradiction": 1},
    },
}

_lock = threading.Lock()

SYSTEM_PROMPT = """You are a research methodologist scoring evidence for a \
systematic appraisal. Score each paper 0-100 on every dimension listed below. \
Every dimension is 0-100; for 偏倚风险 and 矛盾证据 a HIGHER score means MORE \
risk (they are risk dimensions, not quality dimensions).

- relevance 相关性: does it directly answer the question? 0 = off-topic.
- directness 直接性: is it direct experimental evidence, or only a review / \
opinion / inference from other work?
- method 方法可靠性: are the methods appropriate and sufficient for the \
claim (controls, sample size, gold-standard assays, statistics)?
- replication 重复验证: has the finding been reproduced by other groups? Only \
score high if the abstract states independent confirmation; a single-lab \
result with no replication evidence scores low.
- structural 结构支持: is there structural biology evidence (crystal / cryo-EM \
/ docking / predicted structure)? Score 100 only if structure data directly \
supports the claim, 50 if structure exists but is indirect, 0 if none or the \
question does not involve structure — when the paper's topic is unrelated to \
structure, score 50 (neutral).
- data 数据完整性: are the data complete with statistics, error bars, n, and \
raw data availability?
- species 物种匹配: how well does the study organism match the target species.
- bias 偏倚风险: experimental flaws, confounding, small n, selective \
reporting, conflicts of interest, industry funding, overclaiming.
- contradiction 矛盾证据: does this paper's conclusion conflict with the \
established consensus or with other papers in the same batch?

Judge only from the metadata and abstract given. Do not use outside knowledge \
about a specific paper's reputation. Reply ONLY with JSON:
{"papers": [{"id": <int>, "scores": {"<dimension>": <int 0-100>, ...}, "note": "<one short sentence in Chinese, <=60 chars, why the weakest/strongest score>"}]}
Include every id exactly once and every dimension key exactly once per paper."""

CONTRAST_PROMPT = """You are checking a set of retrieved papers for conflicting \
conclusions. Given each paper's id, title and one-line summary, list every pair \
whose conclusions genuinely contradict each other on the same question (not \
merely different topics or different methods). Reply ONLY with JSON:
{"pairs": [{"a": <int>, "b": <int>, "why": "<short Chinese reason>"}]}
Return an empty list when nothing contradicts. Be strict: uncertainty, \
different scopes, or complementary findings are NOT contradictions."""


def dimensions_payload():
    """Dimension metadata for the UI (order matters)."""
    return [{"key": d["key"], "label": d["label"], "hint": d["hint"],
             "direction": d["direction"], "default": d["default"]}
            for d in DIMENSIONS]


def presets_payload():
    return [{"key": k, "label": v["label"], "weights": v["weights"]}
            for k, v in PRESETS.items()]


def default_weights():
    return {d["key"]: d["default"] for d in DIMENSIONS}


def normalize_weights(weights, target_species=""):
    """Clamp user weights to 0-5 ints over the known dimensions.

    物种匹配 only carries information when the user named a target species, so
    it is dropped (weight 0) otherwise instead of silently scoring every paper
    against an unknown organism.
    """
    out = {}
    for key in DIM_KEYS:
        try:
            value = int(round(float((weights or {}).get(key, BY_KEY[key]["default"]))))
        except (TypeError, ValueError):
            value = BY_KEY[key]["default"]
        out[key] = max(0, min(5, value))
    if not (target_species or "").strip():
        out["species"] = 0
    return out


def load_settings():
    """Persisted defaults: {"weights": {...}, "target_species": "..."}."""
    settings = {"weights": default_weights(), "target_species": "", "preset": "balanced"}
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return settings
    if isinstance(saved, dict):
        settings["weights"] = normalize_weights(saved.get("weights"), saved.get("target_species"))
        settings["target_species"] = str(saved.get("target_species") or "")
        settings["preset"] = str(saved.get("preset") or "balanced")
    return settings


def save_settings(weights, target_species="", preset="custom"):
    with _lock:
        data = {"weights": normalize_weights(weights, target_species),
                "target_species": (target_species or "").strip(),
                "preset": preset}
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        atomicio.write_json(SETTINGS_PATH, data, indent=1)
    return data


# ------------------------------------------------------------------ scoring

def impact_score(paper, current_year=None):
    """Citation velocity + publication venue, computed locally.

    Raw citation counts punish young papers, so the citation term uses
    citations per year since publication; the venue term only distinguishes
    "published somewhere" from "no venue recorded" and never encodes a
    hand-made journal ranking.
    """
    year = paper.get("year")
    try:
        year = int(year)
    except (TypeError, ValueError):
        year = None
    now = current_year or date.today().year
    age = max(1, now - year + 1) if year else 3
    cites = paper.get("citations") or 0
    try:
        cites = max(0, int(cites))
    except (TypeError, ValueError):
        cites = 0
    velocity = min(70.0, 20.0 * math.log10(1.0 + cites / age))
    venue = (paper.get("venue") or "").strip()
    venue_pts = 0.0
    if venue:
        venue_pts = 8.0 if re.search(r"arxiv|biorxiv|medrxiv|preprint|ssrn", venue, re.I) else 18.0
    return int(round(min(100.0, velocity + venue_pts)))


def _paper_block(idx, paper, target_species):
    lines = [f"[{idx}] {paper.get('title', '')}"]
    meta = []
    if paper.get("year"):
        meta.append(str(paper["year"]))
    if paper.get("venue"):
        meta.append(str(paper["venue"]))
    if paper.get("citations") is not None:
        meta.append(f"被引 {paper['citations']}")
    if meta:
        lines.append(" / ".join(meta))
    if target_species:
        lines.append(f"用户目标物种: {target_species}")
    abstract = (paper.get("abstract") or "")[:ABSTRACT_CHARS]
    lines.append(f"Abstract: {abstract or '(no abstract)'}")
    return "\n".join(lines)


def _clean_scores(raw):
    """Keep only known dimension keys with int 0-100 values."""
    out = {}
    for key, value in (raw or {}).items():
        if key not in LLM_KEYS:
            continue
        try:
            out[key] = max(0, min(100, int(round(float(value)))))
        except (TypeError, ValueError):
            continue
    return out


def _parse_batch(content, expected_ids):
    """Tolerantly parse {"papers": [{"id", "scores", "note"}]}."""
    try:
        data = llm.parse_json(content)
    except ValueError:
        data = None
    items = []
    if isinstance(data, dict):
        items = data.get("papers") or []
    elif isinstance(data, list):
        items = data
    out = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if idx in expected_ids:
            out[idx] = {"dims": _clean_scores(item.get("scores")),
                        "note": str(item.get("note") or "")[:180]}
    return out


def _score_batch(question, batch, target_species):
    """Score one batch of (index, paper) pairs; returns {index: {...}}."""
    blocks = [_paper_block(idx, p, target_species) for idx, p in batch]
    content = llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nPapers:\n" + "\n\n".join(blocks)},
        ],
        max_tokens=160 + 120 * len(batch),
        temperature=0,
    )
    return _parse_batch(content, {idx for idx, _ in batch})


def detect_contradictions(question, scored):
    """Second pass over the whole set: which papers contradict each other.

    A single abstract cannot reveal that another paper disagrees with it, so
    contradictions are detected across the result set and folded back into the
    矛盾证据 dimension of both papers involved.

    `scored` items are (index, paper, note).
    """
    if len(scored) < 2:
        return []
    lines = [f"[{idx}] {paper.get('title', '')} — {note}"
             for idx, paper, note in scored]
    try:
        content = llm.chat(
            [
                {"role": "system", "content": CONTRAST_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nPapers:\n" + "\n".join(lines)},
            ],
            max_tokens=600,
            temperature=0,
        )
        data = llm.parse_json(content)
    except Exception as e:
        print(f"      [scoring] contradiction pass failed ({e})")
        return []
    pairs = data.get("pairs") if isinstance(data, dict) else data
    out = []
    known = {item[0] for item in scored}
    for item in pairs or []:
        try:
            a, b = int(item["a"]), int(item["b"])
        except (KeyError, TypeError, ValueError):
            continue
        if a in known and b in known:
            out.append({"a": a, "b": b, "why": str(item.get("why") or "")[:160]})
    return out


def grade_of(total):
    if total is None:
        return {"label": "未评分", "cls": "info", "min": None}
    for floor, label, cls in GRADES:
        if total >= floor:
            return {"label": label, "cls": cls, "min": floor}
    return {"label": "弱", "cls": "bad", "min": 0}


def weighted_total(dims, weights):
    """Combine dimensions into one 0-100 score (higher is always better).

    Risk dimensions enter as (100 - score), so turning up 偏倚风险 never makes
    a paper look better.
    """
    num = den = 0.0
    for key, weight in (weights or {}).items():
        if not weight or key not in BY_KEY:
            continue
        score = (dims or {}).get(key)
        if score is None:
            continue
        value = score if BY_KEY[key]["direction"] > 0 else 100 - score
        num += weight * value
        den += weight
    if not den:
        return None
    return int(round(num / den))


def score_papers(question, papers, weights, target_species="", on_progress=None,
                 max_papers=None):
    """Score papers on all ten dimensions.

    Returns (scored_papers, summary): each scored paper is a shallow copy with
    an added "evidence" dict {"dims", "total", "grade", "note", "contradicted_by"}.
    Papers past `max_papers` are returned unscored (evidence=None) instead of
    silently dropped.
    """
    limit = MAX_PAPERS if max_papers is None else int(max_papers)
    weights = normalize_weights(weights, target_species)
    now_year = date.today().year
    # computed locally, kept beside the papers so the caller's dicts are never
    # touched with internal keys
    impacts = {idx: impact_score(p, now_year) for idx, p in enumerate(papers)}
    todo = list(enumerate(papers))[:limit]
    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]

    results = {}
    workers = max(1, min(WORKERS, len(batches))) if batches else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(contextvars.copy_context().run, _score_batch,
                             question, b, target_species): b for b in batches}
        for fut in as_completed(futures):
            batch = futures[fut]
            try:
                results.update(fut.result())
            except Exception as e:
                print(f"      [scoring] batch of {len(batch)} failed ({e})")
            if on_progress:
                try:
                    on_progress(len(results), len(todo))
                except Exception:
                    pass

    scored = []
    for idx, paper in todo:
        entry = results.get(idx) or {}
        dims = dict(entry.get("dims") or {})
        dims["impact"] = impacts[idx]
        scored.append((idx, paper, dims, entry.get("note") or ""))

    # cross-paper contradictions, folded into the risk dimension
    pairs = detect_contradictions(question, [(i, p, n) for i, p, _, n in scored])
    hits = {}
    for pair in pairs:
        for side in (pair["a"], pair["b"]):
            hits[side] = hits.get(side, 0) + 1

    out = []
    for idx, paper, dims, note in scored:
        problems = hits.get(idx, 0)
        if problems:
            base = dims.get("contradiction")
            bumped = min(100, 55 + 15 * problems)
            dims["contradiction"] = bumped if base is None else max(base, bumped)
        # a batch that never answered leaves only locally computed dimensions;
        # reporting a total built from those alone would invent a grade
        judged = any(k in dims for k in LLM_KEYS)
        total = weighted_total(dims, weights) if judged else None
        grade = grade_of(total)
        entry = {
            "dims": dims,
            "total": total,
            "grade": grade["label"],
            "grade_cls": grade["cls"],
            "note": note,
            "contradicted_by": [p["b"] if p["a"] == idx else p["a"] for p in pairs
                                if idx in (p["a"], p["b"])],
            # a batch that failed leaves partial dimensions; say so rather than
            # pretending the total is a full appraisal
            "partial": bool(not judged or len(dims) < len(LLM_KEYS) + 1),
        }
        # evidence_score is the flat copy used for sorting/exporting
        out.append({**paper, "evidence": entry, "evidence_score": entry["total"]})

    for paper in papers[limit:]:
        out.append({**paper, "evidence": None, "evidence_score": None})

    summary = summarize(out)
    summary["weights"] = weights
    summary["target_species"] = (target_species or "").strip()
    summary["max_papers"] = limit
    summary["contradictions"] = len(pairs)
    return out, summary


def summarize(papers):
    """Average / grade distribution over the scored papers."""
    totals = [p["evidence"]["total"] for p in papers
              if p.get("evidence") and p["evidence"].get("total") is not None]
    grades = {}
    for p in papers:
        ev = p.get("evidence")
        if not ev:
            continue
        label = ev.get("grade") or grade_of(ev.get("total"))["label"]
        grades[label] = grades.get(label, 0) + 1
    unscored = sum(1 for p in papers if not p.get("evidence"))
    partial = sum(1 for p in papers
                  if p.get("evidence") and p["evidence"].get("partial"))
    return {
        "scored": len(totals),
        "unscored": unscored,
        "partial": partial,
        "mean": int(round(sum(totals) / len(totals))) if totals else None,
        "grades": grades,
    }


def summarize_label(summary):
    """One-line Chinese summary for chips/logs."""
    if not summary or not summary.get("scored"):
        return "证据评分：无可用评分"
    order = ["强", "较强", "中等", "偏弱", "弱"]
    parts = [f"{g} {summary['grades'][g]}" for g in order if summary.get("grades", {}).get(g)]
    text = f"证据评分 平均 {summary['mean']}（{summary['scored']} 篇"
    if parts:
        text += "：" + " / ".join(parts)
    text += "）"
    if summary.get("unscored"):
        text += f"，另有 {summary['unscored']} 篇未评分"
    return text


if __name__ == "__main__":
    import sys

    print("dimensions:", ", ".join(f"{d['label']}({d['default']})" for d in DIMENSIONS))
    print("weights:", normalize_weights(default_weights()))
    print("impact demo:",
          impact_score({"year": 2023, "citations": 4109, "venue": "arXiv"}),
          impact_score({"year": 2020, "citations": 12, "venue": "Nature"},
                       current_year=2020))
    demo_dims = {k: 70 for k in DIM_KEYS}
    demo_dims.update({"bias": 10, "contradiction": 5, "impact": 60})
    print("weighted total (balanced):", weighted_total(demo_dims, default_weights()))
    print("weighted total (mechanism):",
          weighted_total(demo_dims, normalize_weights(PRESETS["mechanism"]["weights"])))
    q = " ".join(sys.argv[1:]) or "KRAS G12C inhibitor selectivity"
    sample = [{"title": "A covalent KRAS G12C inhibitor", "year": 2023, "citations": 120,
               "venue": "Nature", "abstract": "We report a covalent inhibitor of KRAS G12C with "
               "crystal structures and in vivo efficacy in mouse xenografts."}]
    papers, summary = score_papers(q, sample, default_weights(), "human")
    print(json.dumps(papers[0]["evidence"], ensure_ascii=False, indent=1))
    print(summarize_label(summary))
