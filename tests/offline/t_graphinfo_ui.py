"""点图谱里的圆圈 -> 看到这篇文献本身（本地命中不再请求，其余按需取摘要）。"""
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
ctx.eval("""
// 记录挂在元素上的监听器，这样能真的「点一下画布上的圆圈」
El.prototype.addEventListener = function(t, fn){
  this.__on = this.__on || {}; (this.__on[t] = this.__on[t] || []).push(fn);
};
var __plan = {};                  // 路径 -> 响应（测试里按需改）
function __js(x){ return JSON.parse(JSON.stringify(x)); }
fetch = function(url, opt){
  __fetches.push({url: url, body: (opt || {}).body || ""});
  var body;
  if (url.indexOf("/paperinfo") >= 0) {
    var key = JSON.parse((opt || {}).body || "{}").key;
    body = __plan[key] || {error: "没找到这篇文献"};
    return Promise.resolve({ok: !body.error, json: function(){ return Promise.resolve(__js(body)); }});
  }
  if (url.indexOf("/screening") >= 0) body = {summary: {counts: {}}, decisions: {}};
  else if (url.indexOf("/graph") >= 0) body = {nodes: [], edges: []};
  else body = {};
  return Promise.resolve({ok: true, json: function(){ return Promise.resolve(__js(body)); }});
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

def nd(i, role, doi, title, year, cites, core, sim, support=None):
    n = {"id": i, "role": role, "doi": doi, "title": title, "year": year, "citations": cites,
         "core": core, "sim": sim, "venue": "V-" + title, "authors": ["A" + title, "B" + title]}
    if support is not None: n["support"] = support
    return n

GRAPH_PAYLOAD = {"kind": "pool", "seed": {"id": "pool", "title": "本轮 2 篇"}, "year_range": [2001, 2024],
  "nodes": [nd("W1", "pool", "10.1000/a", "本轮甲", 2024, 10, 1.0, 1.0),
            nd("W2", "pool", "10.1000/b", "本轮乙", 2023, 5, 0.5, 0.4),
            nd("W3", "prior", "10.1000/c", "奠基丙", 2001, 900, 1.0, None, support=2),
            nd("W4", "prior", "10.1000/d", "奠基丁", 2002, 800, 0.9, None, support=2),
            nd("W5", "seed", "", "整轮", None, 0, 1.0, None)],
  "edges": [{"source": "W1", "target": "W3", "kind": "cites", "weight": 3},
            {"source": "W1", "target": "W2", "kind": "coupled", "weight": 4}],
  "core": [], "stats": {"nodes": 5, "edges": 2, "pool": 2, "foundation": 2, "coupled_edges": 1,
                        "citing_edges": 1, "refs_coverage": 2, "seconds": 1.0, "warnings": []}}

ctx.set("__g", json.dumps(GRAPH_PAYLOAD))
ctx.set("__papers", json.dumps([
    {"title": "本轮甲", "doi": "10.1000/a", "arxiv_id": "2401.00001", "year": 2024, "citations": 10,
     "pdf_url": "http://ex/a.pdf", "venue": "V-本轮甲", "authors": ["A本轮甲"],
     "abstract": "MARKER_LOCAL_ABS 本轮工作集里本来就有摘要。", "source": "s2"},
    {"title": "本轮乙", "doi": "10.1000/b", "year": 2023, "citations": 5, "source": "s2", "abstract": ""}]))
ctx.eval("""
current = {id: "run1", mode: "ask", question: "q", papers: JSON.parse(__papers),
           audits: {claims: {verdicts: []}, llm: {verdicts: []}, rules: {lines: []}},
           counts: {}, exports: {}, sort: "relevance"};
GRAPH.kind = "pool"; openGraphPanel();
GRAPH.data = JSON.parse(__g); GRAPH.pos = graphLayout(GRAPH.data, "year"); renderGraphShell();
""")
drain()
check("graph rendered with circles", ctx.eval('$("gmap").innerHTML.indexOf("gnode") > 0'))

def click_node(node_id):
    ctx.eval('(function(){ var els = _get("gmap").__on.mousedown;'
             '  els[0]({target: {closest: function(s){'
             '    return s === ".gnode" ? {dataset: {gid: "' + node_id + '", i: "0"}} : null; }},'
             '    preventDefault: function(){}}); })()')

def info_calls():
    return ctx.eval('__fetches.filter(function(f){ return f.url.indexOf("/paperinfo") >= 0; }).length')

# 1. 本轮论文：工作集里就有，不该发请求 --------------------------------------
click_node("W1"); drain()
detail = ctx.eval('$("gside").innerHTML')
check("clicking the circle opens a paper card", "本轮甲" in detail, detail[:120])
check("the abstract comes from the round's own paper", "MARKER_LOCAL_ABS" in detail, detail[:200])
check("元数据齐全（角色/被引/核心度/刊物/作者）",
      all(x in detail for x in ["🟣 本轮论文", "被引 10", "核心度 1.00", "V-本轮甲", "A本轮甲"]), detail[:300])
check("no /paperinfo call was made for a paper we already have", info_calls() == 0, info_calls())
check("a paper in the round gets the round's own link row",
      'href="https://doi.org/10.1000%2Fa"' in detail and "arxiv.org/abs/2401.00001" in detail and
      "ex/a.pdf" in detail, detail[:400])
check("它有「以它为中心重建图谱」", 'id="gcenter"' in detail)

# 2. 「看它在结果里的卡片」把人送回结果视图并高亮那一张 ----------------------
ctx.eval('$("gcard").__on.click[0]()')
check("back to the results view", "gpanel" not in ctx.eval('$("answer").innerHTML'))
check("and the paper's card is revealed", ctx.eval('_get("cite-1").classList.contains("flash")'))

# 3. 奠基文献：按需取一次摘要 ------------------------------------------------
ctx.eval('__plan = {"10.1000/c": {paper: {title: "奠基丙", doi: "10.1000/c", year: 2001,'
         ' citations: 900, authors: ["C1", "C2"], venue: "V-奠基丙",'
         ' abstract: "MARKER_REMOTE_ABS 从服务器取回来的摘要。",'
         ' pdf_url: "http://ex/c.pdf", openalex_url: "https://openalex.org/W3"}}};')
ctx.eval('GRAPH.kind = "pool"; openGraphPanel(); GRAPH.data = JSON.parse(__g);'
         ' GRAPH.pos = graphLayout(GRAPH.data, "year"); renderGraphShell();')
drain()
click_node("W3")
check("foundation node shows a loading line first",
      "正在取这篇的摘要" in ctx.eval('$("gside").innerHTML'), ctx.eval('$("gside").innerHTML')[:160])
check("exactly one request went out", info_calls() == 1, info_calls())
check("it asked by DOI", json.loads(json.loads(ctx.eval(
      'JSON.stringify(__fetches.filter(function(f){ return f.url.indexOf("/paperinfo") >= 0; })[0].body)')))["key"] == "10.1000/c")
drain()
detail = ctx.eval('$("gside").innerHTML')
check("the fetched abstract lands in the card", "MARKER_REMOTE_ABS" in detail, detail[:200])
check("with the fetched venue/authors and links",
      "V-奠基丙" in detail and "C1" in detail and "openalex.org/W3" in detail and "ex/c.pdf" in detail,
      detail[:400])
check("a foundation node has no 「结果里的卡片」 button",
      'id="gcard"' not in detail)

# 4. 再点一次：关掉 -> 再打开，仍然只请求过一次（按节点缓存） ----------------
click_node("W3"); drain()
check("clicking the selected circle again clears the detail",
      "gdetail" not in ctx.eval('$("gside").innerHTML'))
click_node("W3"); drain()
check("re-opening it uses the cache", info_calls() == 1, info_calls())
check("and the card is there immediately", "MARKER_REMOTE_ABS" in ctx.eval('$("gside").innerHTML'))

# 5. 取不到详情时也要说清楚，并把已有的链接留着 ------------------------------
click_node("W4"); drain()
detail = ctx.eval('$("gside").innerHTML')
check("a 404 shows the server's reason", "没找到这篇文献" in detail, detail[:200])
check("and keeps what the graph itself knows", "奠基丁" in detail and 'href="https://doi.org/10.1000%2Fd"' in detail,
      detail[:300])

print("\nFAILURES: %d" % len(fails))
for f in fails: print("  - " + f)
sys.exit(1 if fails else 0)
