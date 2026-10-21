"""模式切换要把视图切到「这个模式自己的结果」，而不是上一个模式留下的结果。

复现的路径：智能问答提问 -> 切「仅搜索论文」-> 检索 -> 切回「智能问答」，
曾经看到的还是那一列论文。
"""
import json, re, sys
import esprima, quickjs
import _paths  # noqa: F401
from pathlib import Path
import web

js = "\n;\n".join(re.findall(r"<script[^>]*>(.*?)</script>", web.PAGE, re.S))
esprima.parseScript(js)

STUB = Path(__file__).with_name("t_graph_ui.py").read_text(encoding="utf-8").split('STUB = """')[1].split('"""')[0]
ctx = quickjs.Context()
ctx.eval(STUB)
# the real page's radios, so change listeners and :checked can be driven
ctx.eval("""
function __mkMode(v){ var e = new El("mode-" + v); e.value = v; e.checked = (v === "ask");
  e.change = null; e.addEventListener = function(t, fn) { if (t === "change") e.change = fn; };
  return e; }
var __modes = [__mkMode("ask"), __mkMode("search"), __mkMode("explore")];
function __checkedMode(){ for (var i = 0; i < __modes.length; i++) if (__modes[i].checked) return __modes[i]; return __modes[0]; }
document.querySelectorAll = function(sel){ return sel.indexOf('name="mode"') >= 0 ? __modes : []; };
document.querySelector = function(sel){ return sel.indexOf('name="mode"') >= 0 ? __checkedMode() : _get("q"); };
function __pick(v){
  for (var i = 0; i < __modes.length; i++) __modes[i].checked = (__modes[i].value === v);
  __checkedMode().change({target: __checkedMode()});
}
// per-URL answers: the page asks /graph and /screening while rendering
fetch = function(url, opt){
  __fetches.push({url: url, body: (opt || {}).body || ""});
  var body = url.indexOf("/screening") >= 0 ? {summary: {counts: {}}, decisions: {}}
           : url.indexOf("/graph") >= 0 ? {nodes: [], edges: []} : {};
  return Promise.resolve({ok: true, json: function(){ return Promise.resolve(body); }});
};
""")

ctx.eval(js)
ctx.eval("download = __grab; downloadBlob = __grab;")

fails = []
def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + str(extra)))
    if not cond: fails.append(name)

def drain(limit=300):
    n = 0
    while n < limit and ctx.execute_pending_job():
        n += 1
    return n

pw = lambda t, y: {"title": t, "source": "s2", "year": y, "citations": 5, "doi": "10.1000/x" + str(y)}
ASK = {"id": "run-ask", "mode": "ask", "question": "多肽生成", "answer": "## 结论\\n多肽设计的关键是 **MARKER_ANSWER**。",
       "papers": [pw("Peptide paper one", 2024), pw("Peptide paper two", 2023)],
       "audits": {"claims": {"verdicts": []}, "llm": {"verdicts": []}, "rules": {"lines": []}}, "counts": {},
       "exports": {}, "sort": "relevance", "duration_s": 12, "log": "答: line one"}
SEARCH = {"id": "run-search", "mode": "search", "question": "peptide design",
          "answer": "", "papers": [pw("MARKER_PAPER_A", 2022), pw("Search hit two", 2021)],
          "audits": {}, "counts": {}, "exports": {}, "sort": "cited_desc", "duration_s": 7,
          "log": "检索: line one"}

def render(payload):
    # pollJob's done-branch, verbatim minus refreshHistory/rebuildBib
    ctx.set("__p", json.dumps(payload))
    ctx.eval("var __d = JSON.parse(__p);"
             "window.EX = __d.exports || {}; window.SNAP = JSON.stringify(__d);"
             "current = __d; $('sort').value = __d.sort || 'relevance';"
             "renderAudits(__d);")
    drain()

# 1. run an ask question ------------------------------------------------------
render(ASK)
check("ask answer is on screen", "MARKER_ANSWER" in ctx.eval('$("answer").innerHTML'))
check("ask run became the export target", ctx.eval("current.id") == "run-ask", ctx.eval("current.id"))

# 2. switch to 仅搜索论文 before it has ever run ------------------------------
ctx.eval('__pick("search")')
drain()
check("switching to a mode with no result clears the view",
      "MARKER_ANSWER" not in ctx.eval('$("answer").innerHTML'),
      ctx.eval('$("answer").innerHTML')[:120])
check("the cleared view comes back as the empty state", ctx.eval('$("empty").style.display') == "")
check("nothing from the other mode stays as the export target", ctx.eval("current === null"))
check("chips and toolbar are cleared too",
      ctx.eval('$("chips").innerHTML') == "" and ctx.eval('$("toolbar").innerHTML') == "")

# 3. run a search in it, then come back --------------------------------------
render(SEARCH)
check("search list is on screen", "MARKER_PAPER_A" in ctx.eval('$("answer").innerHTML'))
check("search run became the export target", ctx.eval("current.id") == "run-search", ctx.eval("current.id"))
check("the sort box follows the run", ctx.eval('$("sort").value') == "cited_desc", ctx.eval('$("sort").value'))

ctx.eval('__pick("ask")')
drain()
check("going back to 智能问答 shows the ask answer again",
      "MARKER_ANSWER" in ctx.eval('$("answer").innerHTML'), ctx.eval('$("answer").innerHTML')[:120])
check("and no longer the search list", "MARKER_PAPER_A" not in ctx.eval('$("answer").innerHTML'))
check("the ask run is the export target again", ctx.eval("current.id") == "run-ask", ctx.eval("current.id"))
check("chips describe the ask run", "筛选：" in ctx.eval('$("chips").innerHTML') or
      "论文 2 篇" in ctx.eval('$("chips").innerHTML'), ctx.eval('$("chips").innerHTML')[:160])
check("its log line comes back", ctx.eval('$("log").textContent') == "答: line one",
      ctx.eval('$("log").textContent'))
check("the empty state is hidden while a result is shown", ctx.eval('$("empty").style.display') == "none")

# 4. and back again: the search result is still there ------------------------
ctx.eval('__pick("search")')
drain()
check("search keeps its own result", "MARKER_PAPER_A" in ctx.eval('$("answer").innerHTML'))
check("search keeps its own sort", ctx.eval('$("sort").value') == "cited_desc")
check("search keeps its own log", ctx.eval('$("log").textContent') == "检索: line one")

# 5. a run in flight must not have its view yanked away ----------------------
ctx.eval('__pick("ask")'); drain()
ctx.eval("setRunning(true);")
ctx.eval('__pick("search")')
drain()
check("switching modes mid-run leaves the live view alone",
      "MARKER_ANSWER" in ctx.eval('$("answer").innerHTML'))
ctx.eval("setRunning(false);")
ctx.eval('__pick("search")'); drain()
check("once the run is over the switch works again",
      "MARKER_PAPER_A" in ctx.eval('$("answer").innerHTML'))

# 6. 领域探索: the live panel is what that mode remembers --------------------
ctx.eval("EXPLORE.turns = [{directions: [{name: '方向一'}], path: []}]; EXPLORE.last = [{name: '方向一'}];")
ctx.eval("showExplorePanel();")
panel = ctx.eval('$("answer").innerHTML')
check("explore panel is built", "expsurvey" in panel, panel[:120])
ctx.eval('__pick("ask")'); drain()
check("leaving explore shows ask again", "MARKER_ANSWER" in ctx.eval('$("answer").innerHTML'))
ctx.eval('__pick("explore")'); drain()
panel = ctx.eval('$("answer").innerHTML')
check("coming back to 领域探索 rebuilds the panel, not the ask answer",
      "expsurvey" in panel and "MARKER_ANSWER" not in panel, panel[:120])

# 7. a survey report is explore's own result, and outranks the bare panel ----
SURVEY = {"id": "run-survey", "mode": "survey", "question": "多肽", "answer": "## MARKER_SURVEY\\n综述正文",
          "papers": [pw("Survey ref", 2020)],
          "audits": {"claims": {"verdicts": []}, "llm": {"verdicts": []}, "rules": {"lines": []}},
          "counts": {}, "exports": {}, "directions": [{"name": "方向一"}], "duration_s": 40}
render(SURVEY)
check("survey report is on screen", "MARKER_SURVEY" in ctx.eval('$("answer").innerHTML'))
ctx.eval('__pick("search")'); drain()
ctx.eval('__pick("explore")'); drain()
check("explore remembers the report it just produced",
      "MARKER_SURVEY" in ctx.eval('$("answer").innerHTML'), ctx.eval('$("answer").innerHTML')[:120])
check("and it is the export target", ctx.eval("current.id") == "run-survey", ctx.eval("current.id"))

print("\nFAILURES: %d" % len(fails))
for f in fails: print("  - " + f)
sys.exit(1 if fails else 0)
