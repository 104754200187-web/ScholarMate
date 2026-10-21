"""整轮图谱的前端链路：工具栏按钮 -> /poolgraph -> 出图 -> 切回单篇 -> 返回结果."""
import json, re, sys
import esprima, quickjs
from _paths import FIXTURES
from pathlib import Path
import web

js = "\n;\n".join(re.findall(r"<script[^>]*>(.*?)</script>", web.PAGE, re.S))
esprima.parseScript(js)

STUB = Path(__file__).with_name("t_graph_ui.py").read_text(encoding="utf-8").split('STUB = """')[1].split('"""')[0]
ctx = quickjs.Context()
ctx.eval(STUB)
ctx.eval(js)
ctx.eval("download = __grab; downloadBlob = __grab;")

fails = []
def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + str(extra)))
    if not cond: fails.append(name)

def drain(limit=200):
    n = 0
    while n < limit and ctx.execute_pending_job():
        n += 1
    return n

pool = json.load(open(FIXTURES / "pool_graph_real.json", encoding="utf-8"))
seed = json.load(open(FIXTURES / "graph_af2.json", encoding="utf-8"))
if "graph" in seed: seed = seed["graph"]
ctx.set("__pool_raw", json.dumps({"ok": True, "graph": pool}))
ctx.set("__seed_raw", json.dumps({"ok": True, "graph": seed}))
ctx.eval("""
var __pool = JSON.parse(__pool_raw), __seed = JSON.parse(__seed_raw);
var __next = null;
fetch = function (url, opt) {
  __fetches.push({url: url, body: (opt || {}).body || ""});
  var which = url === "/poolgraph" ? __pool : __seed;
  return Promise.resolve({ok: true, json: function () { return Promise.resolve(which); }});
};
function __cgOnly() { return __fetches.filter(function (f) { return f.url === "/poolgraph"; }); }
""")

# 1. the toolbar entry -------------------------------------------------------
ctx.eval("""
var __payload = {id: "run1", mode: "search", question: "q", sort_label: "相关性",
  duration_s: 3, papers: [], exports: {}, retraction: {}};
for (var i = 0; i < 6; i++) {
  __payload.papers.push({title: "P" + i, doi: "10.1000/p" + i, year: 2020 + i,
                         citations: 10 * i, source: "s2", venue: "V", authors: ["A"],
                         abstract: ""});
}
// one preprint with no DOI at all: arXiv id is the only handle it has
__payload.papers[5].doi = "";
__payload.papers[5].arxiv_id = "2401.00001v2";
current = __payload;
renderAudits(__payload);
""")
tb = ctx.eval('$("toolbar").innerHTML')
check("toolbar carries the whole-round entry", 'id="poolgraphBtn"' in tb and "本轮全池图谱" in tb)
check("entry shows the paper count", "（6 篇）" in tb, tb[:120])

# 2. clicking it opens the panel and posts the round -------------------------
h = ctx.eval('(function(){var found=null;var els=[];'
             'var pg = document.getElementById("poolgraphBtn");'
             'return typeof showPoolGraph === "function";})()')
check("showPoolGraph is wired to the button", ctx.eval(
  '(function(){var ok=false;var el=$("poolgraphBtn");'
  'var before=__fetches.length; showPoolGraph(); '
  'return __fetches.length>before;})()') is True)
panel = ctx.eval('$("answer").innerHTML')
check("panel opens in pool mode", 'id="gpanel"' in panel and "本轮论文关系图谱" in panel, panel[:160])
check("progress text explains the wait", "正在拼本轮的引用网络" in ctx.eval('$("gstatus").innerHTML'))
sent = json.loads(ctx.eval("JSON.stringify(__cgOnly())"))
check("exactly one /poolgraph post", len(sent) == 1, len(sent))
body = json.loads(sent[0]["body"])
check("the whole round is sent", len(body["papers"]) == 6, len(body.get("papers", [])))
# arxiv_id belongs in here: it is the only handle S2 can use for a paper with
# no DOI, so the round must carry it. Abstracts / full text / scores must not.
check("only the needed fields are sent",
      set(body["papers"][0]) == {"doi", "title", "year", "citations", "arxiv_id",
                                 "venue", "authors"},
      sorted(body["papers"][0]))
check("no paper ships its abstract",
      all("abstract" not in p and "source" not in p and "evidence" not in p
          for p in body["papers"]))
check("an arXiv-only paper still carries its arXiv id",
      any(p["arxiv_id"] == "2401.00001v2" and not p["doi"] for p in body["papers"]),
      [p for p in body["papers"] if p["arxiv_id"]])

# 3. render ------------------------------------------------------------------
drain()
check("pool payload landed", ctx.eval("GRAPH.kind") == "pool")
check("canvas mounted", 'id="gmap"' in ctx.eval('$("gbody").innerHTML'))
status = ctx.eval('$("gstatus").innerHTML')
check("status speaks pool language",
      ("本轮" in status) and ("共同引用的" in status) and ("参考文献取自" in status), status)
svg = ctx.eval('$("gmap").innerHTML')
check("every node drawn", svg.count("<circle") == len(pool["nodes"]), svg.count("<circle"))
check("every edge drawn", svg.count("<line") == len(pool["edges"]), svg.count("<line"))
side = ctx.eval('$("gside").innerHTML')
for want in ["本轮论文", "共同奠基", "核心文献"]:
    check("side panel lists %s" % want, want in side)
check("no seed-mode wording left in the side panel",
      "后续文献" not in side and "与种子相似度" not in side, side[:200])
check("legend explains floating dots", "没有连线的论文" in side)

# 4. selecting a node --------------------------------------------------------
first = pool["nodes"][0]["id"]
ctx.eval("selectGraphNode(%s)" % json.dumps(first))
detail = ctx.eval('$("gside").innerHTML')
check("detail panel uses pool role wording",
      ("本轮论文" in detail) and ("与种子相似度" not in detail), detail[:200])

# 5. switching to a single paper from inside the pool panel ------------------
ctx.eval('$("gseed").value = "10.1038/s41586-020-2649-2"; graphRebuild();')
drain()
check("switch to a single-paper graph", ctx.eval("GRAPH.kind") == "seed")
check("title follows the mode", "文献关系图谱" in ctx.eval('$("gtitle").textContent'))
check("seed payload rendered", ctx.eval('$("gmap").innerHTML').count("<circle") == len(seed["nodes"]))
check("side panel returns to seed wording", "后续文献" in ctx.eval('$("gside").innerHTML'))

# 6. back --------------------------------------------------------------------
ctx.eval("graphBack();")
back = ctx.eval('$("answer").innerHTML')
check("← 返回结果 restores the list", "本轮全池图谱" in ctx.eval('$("toolbar").innerHTML') and 'id="gpanel"' not in back)

print()
print("FAILURES: %d" % len(fails))
for f in fails: print("  - " + f)
sys.exit(1 if fails else 0)
