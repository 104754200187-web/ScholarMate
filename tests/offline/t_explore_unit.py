"""Offline checks for the exploration branch (no network)."""
import sys
import _paths  # noqa: F401

import explore, graph, verify

fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + str(extra)) if extra and not cond else ""))
    if not cond:
        fails.append(name)

# ---- explore.normalize
raw = {"directions": [
    {"name": "方向A", "en": "topic a", "why": "why a", "queries": ["q a1", "q a2"]},
    {"name": "方向A", "en": "dup", "queries": ["dup"]},              # duplicate name
    {"name": "", "queries": ["nameless"]},                            # no name
    {"name": "只有英文名", "en": "english only"},                       # queries fall back to en
    "not a dict",                                                     # junk
    {"name": "无检索词", "why": "x"},                                  # dropped: no queries/en
]}
dirs = explore.normalize(raw)
check("normalize keeps valid entries", [d["name"] for d in dirs] == ["方向A", "只有英文名"], [d["name"] for d in dirs])
check("normalize assigns ids", [d["id"] for d in dirs] == ["1", "2"])
check("normalize falls back to en as a query", dirs[1]["queries"] == ["english only"])
check("normalize tolerates a non-list", explore.normalize("nope") == [])
check("normalize tolerates a bare list", len(explore.normalize([{"name": "x", "queries": ["q"]}])) == 1)

# ---- survey_queries: round-robin, capped, de-duplicated
cols = [{"name": "a", "queries": ["a1", "a2", "a3"]},
        {"name": "b", "queries": ["b1", "b2", "b3"]},
        {"name": "c", "queries": ["c1", "a1"]}]
qs = explore.survey_queries(cols, cap=6)
check("survey_queries round-robins", qs == ["a1", "b1", "c1", "a2", "b2", "a3"], qs)
check("survey_queries dedupes", explore.survey_queries(cols, cap=10).count("a1") == 1)
check("survey_queries respects the cap", len(explore.survey_queries(cols, cap=4)) == 4)
check("survey_queries falls back to en",
      explore.survey_queries([{"name": "x", "en": "eng term"}], cap=3) == ["eng term"])
check("outline numbers sections", explore.outline(cols).splitlines()[0].startswith("1. a"))
check("names joins", explore.names(cols) == "a、b、c")

# ---- verify: a Chinese references heading must split the answer
answer = ("## 摘要\n\n结论见 [1]，也见 [2]。\n\n"
          "## 参考文献\n\n[1] Real Paper Title (2020)\n[2] Another Real Paper (2021)\n")
body, refs = verify._body(answer)
check("cjk heading splits the body", "参考文献" not in body, body[-40:])
check("cjk refs are parsed", verify.extract_references(answer) == {1: "Real Paper Title", 2: "Another Real Paper"},
      verify.extract_references(answer))
papers = [{"title": "Real Paper Title"}, {"title": "Another Real Paper"}]
items = verify.audit_rules(answer, papers)["items"]
check("cjk citations verify OK", not [i for i in items if i["status"] == "bad"], items)
# the word 参考文献 inside prose must not cut the body
prose = "## 摘要\n\n检索 参考文献 时使用了 [1]。\n\n## 参考文献\n\n[1] Real Paper Title (2020)\n"
b2, _ = verify._body(prose)
check("prose mention of 参考文献 is not a heading", "检索" in b2 and "摘要" in b2, b2)
# an English heading still works
en = "Text [1].\n\nReferences:\n[1] Real Paper Title (2020)\n"
check("english heading still splits", verify._body(en)[1].strip().startswith("[1]"))

# ---- graph routing
check("route: scope", graph.route_entry({"explore_action": "scope"}) == "explore_scope")
check("route: deepen", graph.route_entry({"explore_action": "deepen"}) == "explore_scope")
check("route: survey", graph.route_entry({"explore_action": "survey"}) == "survey_plan")
check("route: chat", graph.route_entry({"chat_message": "hi"}) == "chat_answer")
check("route: ordinary", graph.route_entry({"question": "q"}) == "plan_queries")

d = graph.describe_graph()
ids = [n["id"] for n in d["nodes"]]
check("graph has explore_scope", "explore_scope" in ids)
check("graph has survey_plan", "survey_plan" in ids)
check("explore_scope is terminal", any(e["source"] == "explore_scope" and e["target"] == "__end__" for e in d["edges"]))
check("survey_plan feeds retrieve", any(e["source"] == "survey_plan" and e["target"] == "retrieve" for e in d["edges"]))
check("every node is described", all(n["label"] != n["id"] for n in d["nodes"]),
      [n["id"] for n in d["nodes"] if n["label"] == n["id"]])

# survey needs directions
try:
    graph.run_full("q", mode="survey")
    check("survey without directions raises", False)
except ValueError:
    check("survey without directions raises", True)
except Exception as e:
    check("survey without directions raises", False, type(e).__name__)

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
