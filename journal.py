"""Which venue class a paper comes from: 正刊 / 子刊 / 一区–四区.

The filter exists because two readers of the same search want different floors
of evidence: someone scoping a field wants everything, someone writing a grant
wants 一区 only. So a run can be restricted to any combination of six classes.

Where the answer comes from, in order (anything earlier wins):

1. ``outputs/journal_tiers.json`` -- the table. A person (or an institution's
   分区表 importer) can put an entry here and it is believed. This is the only
   route to an *official* 分区: nothing public is fetched to build one.
2. Built-in rules: the flagship 正刊 list and the 子刊 name families (Nature,
   Science, Cell, Lancet, NEJM, JAMA, BMJ, PNAS series). Read them below -- they
   are name based, so they can be checked by eye rather than trusted.
3. 分区 *estimate*: only when a 区间 box is ticked and neither of the above knows
   the journal, its OpenAlex source record is fetched once (then cached in the
   table) and its 2-year mean citedness is bucketed with ``BANDS``. This is an
   estimate, not 中科院分区: it is recorded as ``src: "openalex"`` so the UI can
   say so, and any table entry overrides it. Replace ``BANDS``/the table with
   the official 分区表 to make the boxes mean the official thing.

Venues that are not journals at all (preprints, conference proceedings) are
recognised by name and never get a 分区: they are "other", and a run keeps them
only when the caller asks for it.
"""

import contextvars
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import atomicio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.environ.get("JOURNAL_TIERS") or os.path.join(BASE_DIR, "outputs", "journal_tiers.json")

# the six boxes, in the order the UI shows them
TIER_KEYS = ("flagship", "sub", "q1", "q2", "q3", "q4")
TIER_LABELS = {"flagship": "正刊", "sub": "子刊", "q1": "一区", "q2": "二区",
               "q3": "三区", "q4": "四区"}

# 正刊: the journal itself is the family's flagship (the 子刊 rule below then
# treats every other journal of the same family as a 子刊). Keep this list short
# and obvious: anything arguable belongs in the table, not here.
FLAGSHIP = {
    "nature", "science", "cell",
    "the lancet", "lancet",
    "new england journal of medicine",
    "jama", "bmj",
    "pnas", "proceedings of the national academy of sciences",
}

# 子刊: a name in one of the flagship families. Matched on the leading words, so
# "Nature Communications", "Nature Reviews Cancer", "npj ..."(*), "Science
# Advances", "The Lancet Oncology", "JAMA Internal Medicine", "BMJ Open",
# "NEJM Evidence", "Cell Reports", "PNAS Nexus" all land here.
# (*) npj titles do not carry the Nature name; add them to the table if needed.
SUB_PREFIXES = (
    "nature", "science", "cell", "the lancet", "lancet", "new england journal of medicine",
    "nejm", "jama", "bmj", "pnas", "proceedings of the national academy of sciences",
)
# Cell Press titles that do not start with "Cell"
SUB_EXTRA = {
    "cancer cell", "molecular cell", "cell stem cell", "cell reports", "cell metabolism",
    "cell host microbe", "cell host & microbe", "developmental cell", "immunity",
    "neuron", "current biology", "molecular cell", "structure", "chemistry biology",
    "trends in biochemical sciences", "trends in cell biology", "trends in ecology evolution",
    "trends in neurosciences", "trends in pharmacological sciences",
}

# an estimate, not an official 分区: bands on OpenAlex 2-year mean citedness.
# Values chosen so the obvious cases land where a reader expects them (Nature
# 19.3 / Science 21.5 / Cell 41.4 / Nature Communications 18.3 -> 一区;
# Scientific Reports 5.2 / PLOS ONE 3.4 -> 三区). Editable in the table file.
BANDS = {"q1": 15.0, "q2": 8.0, "q3": 3.0}

_PREPRINT_RE = re.compile(
    r"arxiv|biorxiv|medrxiv|chemrxiv|techrxiv|ssrn|research square|zenodo|osf\.io|"
    r"preprint|posted-content|preprints\.org|hal open", re.I)
_CONFERENCE_RE = re.compile(
    r"proceedings|conference|symposium|workshop|congress|colloquium|annual meeting|"
    r"meeting of the|^proc\.|neurips|nips|icml|iclr|cvpr|iccv|eccv|aaai|ijcai|"
    r"acl\b|emnlp|naacl|coling|sigir|kdd|www\b|chi\b|ubicomp|sigcomm|"
    r"^cs\.|^cond-mat|^math\.|^q-bio|^physics\.|^stat\.|^eess\.|^econ\.", re.I)
# PNAS publishes "Proceedings ..." but it is a journal, not a conference.
_PNAS_RE = re.compile(r"^proceedings of the national academy of sciences\b|^pnas\b", re.I)

_WS_RE = re.compile(r"[^a-z0-9]+")
# Google Scholar (SerpAPI) hands the venue back exactly as Google renders it, and
# Google cuts long ones off with an ellipsis ("Advanced Energy …").
_TRUNCATED_RE = re.compile(r"…|\.\.\.")


def norm_name(name):
    """Fold a journal name for table lookup ("PLoS ONE" == "PLOS ONE")."""
    s = (name or "").lower()
    s = re.sub(r"\s*\([^)]*\)", " ", s)          # drop "(Cornell University)" and friends
    s = re.sub(r"^the\s+", "", s.strip())
    s = re.sub(r"\s*\b(springer|elsevier|wiley|mdpi|frontiers media)\b\s*$", "", s)
    return _WS_RE.sub(" ", s).strip()


def _shown_name(venue):
    """The venue string as a reader would write it (drops the noise suffixes)."""
    s = re.sub(r"\s*\([^)]*\)", "", (venue or "").strip())
    return re.sub(r"\s+", " ", s).strip(" .;,")


def kind_of(venue):
    """journal / preprint / conference / unknown -- by name, no network."""
    v = (venue or "").strip()
    if not v:
        return "unknown"
    if _PREPRINT_RE.search(v):
        return "preprint"
    if _PNAS_RE.search(v):
        return "journal"
    if _CONFERENCE_RE.search(v):
        return "conference"
    return "journal"


def flagship_of(venue):
    n = norm_name(venue)
    if not n:
        return False, False
    if n in FLAGSHIP:
        return True, False
    if n.startswith("proceedings of the national academy of sciences"):
        # The official name often carries "of the United States of America".
        return True, False
    for pre in SUB_PREFIXES:
        if n.startswith(pre) and n != pre:
            # "Science Bulletin" is not an AAAS 子刊, but "Science Advances" is;
            # the family words below are the ones a reader would accept.
            if pre in ("science", "cell") and n.split()[1:2] and n.split()[1] not in (
                    "advances", "translational", "immunology", "robotics", "signaling",
                    "reports", "metabolism", "stem", "host", "chemical", "systems", "reviews"):
                continue
            return False, True
    if n in SUB_EXTRA:
        return False, True
    return False, False


def quartile_from_citedness(value, bands=None):
    """Bucket an OpenAlex 2-year mean citedness into 1..4 (None when unknown)."""
    if not value or value <= 0:
        return None
    b = dict(BANDS if bands is None else bands)
    if value >= b.get("q1", 15.0):
        return 1
    if value >= b.get("q2", 8.0):
        return 2
    if value >= b.get("q3", 3.0):
        return 3
    return 4


# --------------------------------------------------------------- the table

_table = None
# Reentrant: the accessors below (table/_entries) take this lock, and
# save_table/_learn/set_entry call those accessors while already holding it.
_table_lock = threading.RLock()
_flush_lock = threading.Lock()
_dirty = False
_last_flush = 0.0
FLUSH_S = float(os.environ.get("JOURNAL_FLUSH_S", "10"))


def _empty_table():
    return {"source": "built-in rules + OpenAlex 2yr mean citedness estimate",
            "updated": "", "bands": dict(BANDS), "journals": {}}


def table():
    global _table
    with _table_lock:
        if _table is None:
            try:
                with open(TABLE_PATH, encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict) or not isinstance(loaded.get("journals"), dict):
                    raise ValueError("bad table")
                loaded.setdefault("bands", dict(BANDS))
                _table = loaded
            except Exception:
                _table = _empty_table()
        return _table


def forget():
    """Drop the in-memory table (tests, or after rewriting the file by hand)."""
    global _table
    with _table_lock:
        _table = None


def _entries():
    return table()["journals"]


def entry_for(venue):
    """The table entry for a venue, or None. Exact name first, then the folded
    name, so "PLoS ONE" and "PLOS ONE" find the same row."""
    journals = _entries()
    if not journals:
        return None
    if venue in journals:
        return journals[venue]
    n = norm_name(venue)
    if n and n in journals:
        return journals[n]
    return None


def _learn(venue, entry):
    """Add or refine a table row. A user row is never overwritten."""
    global _dirty
    n = norm_name(venue) or venue
    if not n:
        return
    with _table_lock:
        journals = _entries()
        old = journals.get(n)
        if old and old.get("src") == "user":
            return
        row = dict(entry)
        row["name"] = _shown_name(venue)
        row["src"] = row.get("src") or "openalex"
        journals[n] = row
        _dirty = True


def save_table(force=False):
    """Write the table back, merging whatever is on disk first.

    Two runs can learn about journals at the same time, and both write this one
    file; merging before writing is what keeps one run from dropping the other
    run's new rows (the file is written atomically either way).
    """
    global _dirty, _last_flush
    with _flush_lock:
        if not _dirty and not force:
            return 0
        now = time.time()
        if not force and now - _last_flush < FLUSH_S:
            return 0
        with _table_lock:
            mine = dict(_entries())
            bands = dict(table().get("bands") or BANDS)
        try:
            with open(TABLE_PATH, encoding="utf-8") as f:
                disk = json.load(f)
            existing = disk.get("journals") if isinstance(disk.get("journals"), dict) else {}
        except Exception:
            disk, existing = {}, {}
        merged = dict(existing)
        added = 0
        for k, v in mine.items():
            if k not in merged:
                added += 1
            elif merged[k].get("src") == "user" and v.get("src") != "user":
                continue        # never replace a user row
            merged[k] = v
        out = {"source": disk.get("source") or table().get("source") or "",
               "updated": time.strftime("%Y-%m-%d"),
               "bands": disk.get("bands") or bands,
               "journals": merged}
        try:
            os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
            atomicio.write_json(TABLE_PATH, out)
        except OSError:
            return 0
        _dirty = False
        _last_flush = now
        return added


def set_entry(venue, tier=None, flagship=None, sub=None, note=""):
    """Record a hand-made (or imported) row: this is the official-data seam."""
    if not norm_name(venue):
        return None
    row = {"tier": tier, "flagship": flagship, "sub": sub, "note": note, "src": "user"}
    with _table_lock:
        _entries()[norm_name(venue)] = row
        row["name"] = _shown_name(venue)
    globals()["_dirty"] = True
    save_table(force=True)
    return row


# ------------------------------------------------------- the OpenAlex probe

_lookup_lock = threading.Lock()
_looked_up = set()          # journal names already asked about in this process
_lookup_failures = 0
SOURCE_URL = "https://api.openalex.org/sources"
LOOKUP_TIMEOUT = float(os.environ.get("JOURNAL_LOOKUP_TIMEOUT", "12"))


def _openalex_source(venue):
    """The OpenAlex source record for a journal name, or None.

    Returns (display_name, source_type, 2yr_mean_citedness).
    """
    global _lookup_failures
    import search                       # imported late: search imports this module                       # imported late: search imports this module
    params = {"filter": "display_name.search:" + re.sub(r"[,:]", " ", _shown_name(venue)),
              "per-page": 5}
    search.openalex_auth(params)
    try:
        data = search.http_get_json(SOURCE_URL, params, timeout=LOOKUP_TIMEOUT, retries=1)
    except Exception:
        _lookup_failures += 1
        return None
    target = norm_name(venue)
    # a success clears the streak: the budget above is for a source that is
    # down, not for a network that drops the occasional request
    _lookup_failures = 0
    best = None
    for res in data.get("results") or []:
        got = norm_name(res.get("display_name"))
        if got and got == target:
            best = res
            break
        # the containment fallback is for suffixes ("... International Edition",
        # "(New York, N.Y.)"), so it needs a whole name: on a short fragment it
        # matches any journal containing it
        if (not best and got and len(got) >= 8 and len(target) >= 8
                and (target in got or got in target)):
            best = res
    if best is None:
        return None
    stats = best.get("summary_stats") or {}
    return (best.get("display_name") or "", best.get("type") or "",
            stats.get("2yr_mean_citedness") or 0.0)


def stats():
    return {"looked_up": len(_looked_up), "failures": _lookup_failures,
            "rows": len(_entries())}


LOOKUP_WORKERS = int(os.environ.get("JOURNAL_LOOKUP_WORKERS", "8"))


def resolve_many(venues):
    """Classify several venue names, looking up the unresolved ones in parallel.

    Parallel because a run can touch dozens of new journals at once and the
    filter sits inside retrieval: one batched pass costs about as long as the
    slowest single lookup, a pass per paper would cost the sum.
    """
    names = [v for v in dict.fromkeys(_shown_name(v) for v in (venues or [])) if v]
    if len(names) <= 1:
        return {n: classify(n) for n in names}
    out = {}
    with ThreadPoolExecutor(max_workers=max(1, min(LOOKUP_WORKERS, len(names)))) as ex:
        for name, info in zip(names, ex.map(classify, names)):
            out[name] = info
    return out


# -------------------------------------------------------------- classifying

def classify(venue, allow_lookup=True):
    """The venue's classes, as a plain dict.

    Never raises: a paper whose journal cannot be identified is classified
    "other", which is what makes the drop count in the UI meaningful rather
    than a silent zero.
    """
    shown = _shown_name(venue)
    kind = kind_of(shown)
    out = {"venue": shown, "kind": kind, "flagship": False, "sub": False,
           "quartile": None, "src": "", "citedness": None}
    if kind == "unknown":
        return out
    if kind == "journal" and _TRUNCATED_RE.search(shown):
        # A fragment cannot identify the journal, and a fuzzy OpenAlex lookup on
        # one would confidently answer with a *different* journal's impact
        # ("Advanced …" -> Advanced Materials), which is worse than no answer.
        # So it stays unclassified: kept only when 保留未分类 is ticked.
        out["kind"] = "unknown"
        return out
    row = entry_for(shown)
    if not row and kind in ("preprint", "conference"):
        return out
    # The rules and the table answer different questions -- the rules know the
    # 正刊/子刊 families, the table holds the 分区 -- so a row overrides a field
    # only when it states one. set_entry("X", tier=2) therefore keeps "X" a
    # 子刊 and just moves its 分区; set_entry("X", sub=False) takes that back.
    fl, sub = (False, False) if kind != "journal" else flagship_of(shown)
    if row:
        if row.get("flagship") is not None:
            fl = bool(row.get("flagship"))
        if row.get("sub") is not None:
            sub = bool(row.get("sub"))
    out["flagship"], out["sub"] = fl, sub
    if row:
        tier = row.get("tier")
        try:
            out["quartile"] = int(tier) if tier else None
        except (TypeError, ValueError):
            out["quartile"] = None
        out["citedness"] = row.get("citedness")
        out["src"] = row.get("src") or "table"
    if fl or sub:
        out["src"] = out["src"] or "rule"
    if row and (fl or sub or out["quartile"]):
        return out
    if kind != "journal":
        return out
    if out["quartile"] is None and allow_lookup:
        with _lookup_lock:
            already = shown in _looked_up
            _looked_up.add(shown)
        if not already:
            got = _openalex_source(shown)
            if got:
                name, stype, cited = got
                if (stype or "").lower() == "journal" or not stype:
                    q = quartile_from_citedness(cited, table().get("bands"))
                    _learn(shown, {"tier": q, "flagship": fl, "sub": sub,
                                   "citedness": round(cited, 2), "src": "openalex"})
                    out["quartile"], out["citedness"] = q, round(cited, 2)
                    out["src"] = "openalex"
                elif (stype or "").lower() == "conference":
                    out["kind"] = "conference"
        else:
            row = entry_for(shown)
            if row:
                try:
                    out["quartile"] = int(row.get("tier")) if row.get("tier") else None
                except (TypeError, ValueError):
                    pass
                out["citedness"] = row.get("citedness")
                out["src"] = row.get("src") or out["src"]
    if out["quartile"] is not None and not out["src"]:
        out["src"] = "table"
    return out


_hinted = False


def _hint_once():
    """Say once, in the run's log, why 分区 filtering is running blind."""
    global _hinted
    if _hinted:
        return
    _hinted = True
    print("      [tier] 分区数据源连续失败，本次运行不再逐刊查询：未识别的刊物按「未分类」处理"
          "（可勾选「保留会议/预印本/未分类」把它们留下）")


def classes_of(info):
    """The tier keys a classification belongs to (used by the filter)."""
    keys = []
    if info.get("flagship"):
        keys.append("flagship")
    if info.get("sub"):
        keys.append("sub")
    q = info.get("quartile")
    if q:
        keys.append("q%d" % q)
    return keys


# ------------------------------------------------------------- run-scoped filter

_current = contextvars.ContextVar("journal_tier_filter", default=None)


def current_filter():
    """The tier filter of the run executing in this thread, or None."""
    return _current.get()


def set_filter(flt):
    return _current.set(flt)


def reset_filter(token):
    _current.reset(token)


class TierFilter:
    """One run's venue-class filter, plus the tally the UI reports.

    A run-scoped object rather than module state, because two runs can share the
    process: each has its own boxes and its own count of what it dropped. The
    classification table behind it *is* shared, which is the point -- the second
    run starts already knowing every journal the first one looked up.
    """

    def __init__(self, tiers, keep_others=False, allow_lookup=True):
        self.tiers = [t for t in (tiers or []) if t in TIER_KEYS]
        self.keep_others = bool(keep_others)
        self.allow_lookup = bool(allow_lookup)
        self.enabled = bool(self.tiers)
        self.max_lookup_failures = int(os.environ.get("JOURNAL_MAX_LOOKUP_FAILURES", "4"))
        self._lock = threading.Lock()
        self._stats = {"kept": 0, "dropped": 0, "other": 0, "estimated": 0}

    # -- the filter itself -------------------------------------------------
    def keep(self, paper):
        if not self.enabled:
            return True
        info = classify((paper or {}).get("venue") or "", allow_lookup=self.allow_lookup)
        keys = classes_of(info)
        hit = any(k in self.tiers for k in keys)
        # a journal can be 一区 by estimate and still be the reader's chosen
        # class; remembering that it was estimated is what the UI marks
        estimated = hit and info.get("src") == "openalex"
        # keep_others is the escape hatch for what the boxes cannot describe: a
        # venue with no class at all (会议/预印本/未分类). A journal that *is*
        # classified, but into an unticked box, is still out.
        kept = hit or (self.keep_others and not keys)
        with self._lock:
            if kept:
                self._stats["kept"] += 1
                if estimated:
                    self._stats["estimated"] += 1
            else:
                self._stats["dropped"] += 1
                if not keys:
                    self._stats["other"] += 1
        return kept

    def filter(self, papers):
        """Keep the papers whose venue class is one of the ticked boxes.

        The venues of the whole batch are resolved together (see
        resolve_many), so a run pays for each new journal at most once.
        """
        if not self.enabled:
            return papers
        # a source that is down (or a proxy having a bad day) must not turn the
        # filter into a per-paper timeout: after a few failures the rest of the
        # run stops asking and treats unknown journals as unclassified
        if self.allow_lookup and _lookup_failures >= self.max_lookup_failures:
            self.allow_lookup = False
            _hint_once()
        if self.allow_lookup and any(t.startswith("q") for t in self.tiers):
            resolve_many([(p or {}).get("venue") or "" for p in papers])
        return [p for p in papers if self.keep(p)]

    # -- reporting ---------------------------------------------------------
    def summary(self):
        save_table()
        with self._lock:
            s = dict(self._stats)
        return {
            "enabled": self.enabled, "tiers": list(self.tiers),
            "labels": [TIER_LABELS[t] for t in self.tiers],
            "keep_others": self.keep_others,
            "kept": s["kept"], "dropped": s["dropped"],
            "dropped_unclassified": s["other"], "estimated": s["estimated"],
            "seen": s["kept"] + s["dropped"],
        }


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args and args[0] == "check":
        for v in args[1:]:
            print(json.dumps(classify(v), ensure_ascii=False))
    else:
        print(json.dumps({"table": TABLE_PATH, "bands": table().get("bands"),
                          "rows": len(_entries()), "stats": stats()},
                         ensure_ascii=False, indent=1))
