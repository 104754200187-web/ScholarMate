"""Venue classes and the 正刊/子刊/一区-四区 filter -- offline (lookups stubbed)."""
import json, os, shutil, sys, tempfile
import _paths  # noqa: F401
import journal

fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  " + str(extra)))
    if not cond: fails.append(name)

TMPDIR = tempfile.mkdtemp(prefix="sm_tier_unit_")
TABLE = os.path.join(TMPDIR, "tier_unit.json")
journal.TABLE_PATH = TABLE
if os.path.exists(TABLE):
    os.remove(TABLE)
journal.forget()
# no network in this test: a lookup returns nothing, so classes come from the
# rules and the table only -- which is exactly what we want to pin down
journal._openalex_source = lambda venue: None
journal._looked_up = set()
journal.save_table(force=True)

# ---- 1. the built-in rules ------------------------------------------------
cases = [("Nature", "flagship"), ("Science", "flagship"), ("Cell", "flagship"),
         ("The Lancet", "flagship"), ("New England Journal of Medicine", "flagship"),
         ("PNAS", "flagship"), ("JAMA", "flagship"), ("BMJ", "flagship"),
         ("Proceedings of the National Academy of Sciences", "flagship"),
         ("Proceedings of the National Academy of Sciences of the United States of America",
          "flagship"),
         ("Nature Communications", "sub"), ("Nature Reviews Cancer", "sub"),
         ("Science Advances", "sub"), ("Cell Reports", "sub"),
         ("The Lancet Oncology", "sub"), ("JAMA Network Open", "sub"),
         ("BMJ Open", "sub"), ("NEJM Evidence", "sub"), ("PNAS Nexus", "sub"),
         ("Cancer Cell", "sub")]
for name, want in cases:
    info = journal.classify(name, allow_lookup=False)
    got = "flagship" if info["flagship"] else ("sub" if info["sub"] else "none")
    check(f"rule: {name} -> {want}", got == want, got)
check("flag is not a 子刊 of anything", journal.classify("Physical Review Letters",
                                                        allow_lookup=False)["sub"] is False)

# ---- 2. non-journals ------------------------------------------------------
kinds = {"arXiv (Cornell University)": "preprint", "bioRxiv": "preprint",
         "Proceedings of the  …": "conference", "NeurIPS": "conference",
         "Proceedings of the National Academy of Sciences": "journal",
         "PNAS Nexus": "journal",
         "cs.CL": "conference", "": "unknown", "Zenodo (CERN...)": "preprint"}
for v, want in kinds.items():
    got = journal.classify(v, allow_lookup=False)["kind"]
    check(f"kind: {v or '<empty>'} -> {want}", got == want, got)

# the venue Google Scholar hands back is cut off with "…"; it must not be
# guessed at, or a fragment like "Advanced …" would answer with the impact of
# whichever journal OpenAlex returns first
check("a truncated venue is not looked up",
      journal.classify("Advanced Energy …", allow_lookup=False)["kind"] == "unknown")
check("and it lands in no tier class",
      journal.classes_of(journal.classify("Advanced Energy …", allow_lookup=False)) == [])
check("a truncated 会议 name is still a 会议",
      journal.classify("Proceedings of the  …", allow_lookup=False)["kind"] == "conference")

# ---- 3. bands -------------------------------------------------------------
for cite, want in ((20, 1), (15, 1), (12, 2), (8, 2), (5, 3), (3, 3), (1.2, 4), (0, None)):
    got = journal.quartile_from_citedness(cite)
    check(f"band {cite} -> q{want}", got == want, got)

# ---- 4. the table wins over the rules and over the estimate ---------------
journal.set_entry("Scientific Reports", tier=3)
journal.forget()
info = journal.classify("Scientific Reports", allow_lookup=False)
check("table 分区 is used", info["quartile"] == 3, info)
check("table rows are marked as such", info["src"] == "user", info["src"])
journal.set_entry("Nature Communications", tier=2)     # an override the user may want
journal.forget()
check("an override can move a 子刊's 分区",
      journal.classify("Nature Communications", allow_lookup=False)["quartile"] == 2)
check("the table survives a reload from disk",
      len(json.load(open(TABLE))["journals"]) >= 2)
# a row may also take a class away, which is the point of stating False
journal.set_entry("Science Advances", sub=False)
journal.forget()
check("sub=False takes the 子刊 class back",
      journal.classify("Science Advances", allow_lookup=False)["sub"] is False)

# ---- 5. the filter --------------------------------------------------------
papers = [{"venue": "Nature", "title": "a"}, {"venue": "Scientific Reports", "title": "b"},
          {"venue": "Nature Communications", "title": "c"}, {"venue": "arXiv", "title": "d"},
          {"venue": "", "title": "e"}]
def kept(tiers, keep_others=False):
    flt = journal.TierFilter(tiers, keep_others, allow_lookup=False)
    return [p["title"] for p in flt.filter(papers)], flt.summary()
check("一区 keeps only the journal the table puts in 一区 (none here)",
      kept(["q1"])[0] == [], kept(["q1"])[0])
check("二区 keeps the overridden 子刊", kept(["q2"])[0] == ["c"], kept(["q2"])[0])
check("三区 keeps Scientific Reports", kept(["q3"])[0] == ["b"], kept(["q3"])[0])
check("子刊 keeps only the family title","" or kept(["sub"])[0] == ["c"], kept(["sub"])[0])
check("正刊 keeps only Nature", kept(["flagship"])[0] == ["a"], kept(["flagship"])[0])
check("boxes combine (OR)", kept(["flagship", "q3"])[0] == ["a", "b"], kept(["flagship", "q3"])[0])
titles, summary = kept(["q3"])
check("the tally counts what was dropped", summary["kept"] == 1 and summary["dropped"] == 4, summary)
check("and how many had no class at all", summary["dropped_unclassified"] == 2, summary)
check("summary carries the labels", summary["labels"] == ["三区"], summary["labels"])
titles, summary = kept(["q3"], keep_others=True)
check("keep_others keeps only the unclassified rest (会议/预印本/未分类)",
      titles == ["b", "d", "e"], titles)
check("a classified journal in an unticked box stays out",
      "a" not in titles and "c" not in titles, titles)
check("and the tally says so",
      summary["kept"] == 3 and summary["dropped"] == 2
      and summary["dropped_unclassified"] == 0, summary)
flt = journal.TierFilter([], allow_lookup=False)
check("no boxes = no filtering at all", flt.filter(papers) == papers and flt.enabled is False)
check("an unknown box name is ignored", journal.TierFilter(["q9", "bogus"], allow_lookup=False).tiers == [])

# ---- 6. two runs do not share boxes or tallies ----------------------------
a = journal.TierFilter(["q3"], allow_lookup=False)
b = journal.TierFilter(["flagship"], allow_lookup=False)
a.keep({"venue": "Scientific Reports"}); a.keep({"venue": "Nature"})
b.keep({"venue": "Nature"})
check("tallies are per run", a.summary()["kept"] == 1 and b.summary()["kept"] == 1,
      (a.summary(), b.summary()))

# ---- 7. a two-writer merge does not lose rows -----------------------------
journal.forget()
journal._learn("Journal A", {"tier": 1})
journal.save_table(force=True)
journal.forget()
journal._learn("Journal B", {"tier": 2})
journal.save_table(force=True)
rows = json.load(open(TABLE))["journals"]
check("merging kept both writers' rows", "journal a" in rows and "journal b" in rows, list(rows))

print()
print("FAILURES: %d" % len(fails))
shutil.rmtree(TMPDIR, ignore_errors=True)
sys.exit(1 if fails else 0)
