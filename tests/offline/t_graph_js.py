"""Exercise the文献关系图谱 front-end (layout, render, side panel, CSV) in quickjs."""
import json, re, sys
import esprima, quickjs
from _paths import FIXTURES
import web

js = "\n;\n".join(re.findall(r"<script[^>]*>(.*?)</script>", web.PAGE, re.S))
esprima.parseScript(js)
print("esprima: page JS parses OK")

STUB = """
function El(id){this.id=id;this.innerHTML="";this.textContent="";this.value="";this.style={};
 this.classList={add:function(){},remove:function(){},toggle:function(){},contains:function(){return false;}};
 this.children=[];this.disabled=false;this.open=false;this.checked=false;this.dataset={};this.hidden=false;}
El.prototype.addEventListener=function(){};El.prototype.appendChild=function(c){return c;};
El.prototype.closest=function(){return null;};
El.prototype.querySelector=function(s){return s.charAt(0)==="#"?_get(s.slice(1)):null;};
El.prototype.querySelectorAll=function(){return [];};El.prototype.scrollIntoView=function(){};
El.prototype.focus=function(){};El.prototype.remove=function(){};
El.prototype.setAttribute=function(k,v){this[k]=v;};El.prototype.getAttribute=function(k){return this[k]||null;};
El.prototype.cloneNode=function(){var e=new El(this.id);e.innerHTML=this.innerHTML;
  e.querySelector=function(){return this.__sub||(this.__sub=new El("clonesub"));};
  __lastClone=e;return e;};
El.prototype.getBoundingClientRect=function(){return {left:0,top:0,width:960,height:640};};
var _els={};function _get(id){if(!id)return null;if(!_els[id])_els[id]=new El(id);return _els[id];}
var document={getElementById:_get,createElement:function(t){return new El(t);},
 querySelector:function(){return _get("q");},querySelectorAll:function(){return [];},
 addEventListener:function(){},body:new El("body"),documentElement:new El("html")};
var window={addEventListener:function(){},matchMedia:function(){return {matches:false,addEventListener:function(){}};},
 getSelection:function(){return {length:0,toString:function(){return "";}};},print:function(){},scrollTo:function(){},location:{hash:""}};
var location=window.location;var navigator={clipboard:{writeText:function(){return Promise.resolve();}},userAgent:"t"};
var localStorage={getItem:function(){return null;},setItem:function(){},removeItem:function(){}};
var setTimeout=function(){return 0;};var setInterval=function(){return 0;};var clearTimeout=function(){};
var requestAnimationFrame=function(){return 0;};
var fetch=function(){return Promise.resolve({ok:true,json:function(){return Promise.resolve({});}});};
var console={log:function(){},error:function(){},warn:function(){}};var alert=function(){};
var URL={createObjectURL:function(){return "blob:x";},revokeObjectURL:function(){}};
var Blob=function(parts){this.parts=parts||[];};var FileReader=function(){};
var __cap=null;function __grab(n,c,m){__cap={n:n,c:c,m:m};}
"""

ctx = quickjs.Context()
ctx.eval(STUB)
ctx.eval(js)
ctx.eval("download = __grab;")
ctx.eval("downloadBlob = __grab;")

fails = []
def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name + ("" if cond else "  <- " + str(extra)))
    if not cond: fails.append(name)

g = json.load(open(FIXTURES / "graph_af2.json", encoding="utf-8"))
if "graph" in g: g = g["graph"]
ctx.set("__G_raw", json.dumps(g))
ctx.eval("var __G = JSON.parse(__G_raw);")

# --- pure helpers -----------------------------------------------------------
ctx.eval('var __seed = graphSeedOf({doi:"10.1/x", title:"T"});')
check("graphSeedOf prefers DOI", ctx.eval("__seed") == "10.1/x", ctx.eval("__seed"))
ctx.eval('var __seed2 = graphSeedOf({title:"Only Title"});')
check("graphSeedOf falls back to title", ctx.eval("__seed2") == "Only Title", ctx.eval("__seed2"))
ctx.eval('var __row = graphRowHTML({doi:"10.1/x", title:"T"});')
check("graphRowHTML emits button+seed", "data-gseed=" in ctx.eval("__row") and "关系图谱" in ctx.eval("__row"))
check("graphRowHTML empty without handle", ctx.eval('graphRowHTML({})') == "")

c0 = ctx.eval("yearColor(0)"); c1 = ctx.eval("yearColor(1)")
check("yearColor earliest is blue (hue 220)", c0 == "hsl(220,72%,52%)", c0)
check("yearColor latest is red (hue 20)", c1 == "hsl(20,72%,52%)", c1)

# --- layout -----------------------------------------------------------------
ctx.eval("GRAPH.data = __G; GRAPH.pos = graphLayout(GRAPH.data, 'year');")
W = ctx.eval("GRAPH.pos.W"); H = ctx.eval("GRAPH.pos.H")
check("graphLayout W/H", (W, H) == (960, 640), (W, H))
n = ctx.eval("GRAPH.pos.pos.length"); nnode = len(g["nodes"])
check("graphLayout keeps every node", n == nnode, (n, nnode))
ctx.eval("var __inb = GRAPH.pos.pos.every(function(p){return p.x>=20&&p.x<=GRAPH.pos.W-20&&p.y>=20&&p.y<=GRAPH.pos.H-20;});")
check("all nodes inside canvas", ctx.eval("__inb") is True)
ctx.eval("var __seedctr = (function(){var i=GRAPH.data.nodes.findIndex(function(x){return x.role=='seed';});"
         "var p=GRAPH.pos.pos[i];return Math.abs(p.x-480)<1 && Math.abs(p.y-320)<1;})();")
check("seed is pinned to centre", ctx.eval("__seedctr") is True)
ctx.eval("var __side = (function(){var pr=[],de=[];GRAPH.data.nodes.forEach(function(x,i){"
         "(x.role=='prior'?pr:de).push(GRAPH.pos.pos[i].x);});"
         "var avg=function(a){return a.reduce(function(s,v){return s+v;},0)/a.length;};"
         "return {p:avg(pr),d:avg(de)};})();")
side = ctx.eval("JSON.stringify(__side)"); side = json.loads(side)
check("prior left of centre, derivative right", side["p"] < 480 < side["d"], side)
# year axis: does vertical position actually encode publication year?
ctx.eval("""function __yeartrend(mode) {
  var L = graphLayout(G_LAST, mode), yr = G_LAST.year_range;
  var xs = [], ys = [];
  G_LAST.nodes.forEach(function (n, i) { xs.push((n.year || yr[0]) - yr[0]); ys.push(L.pos[i].y); });
  var n = xs.length, mx = 0, my = 0, i;
  for (i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
  mx /= n; my /= n;
  var sxy = 0, sxx = 0, syy = 0;
  for (i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) * (xs[i] - mx);
    syy += (ys[i] - my) * (ys[i] - my);
  }
  return sxy / Math.sqrt(sxx * syy);
}""")
ctx.eval("var G_LAST = GRAPH.data;")
ctx.eval("GRAPH.mode='year'; var __cy = __yeartrend('year'), __cs = __yeartrend('sim');")
cy = ctx.eval("__cy"); cs = ctx.eval("__cs")
print("   corr(year, y): year mode %+.3f | sim mode %+.3f" % (cy, cs))
check("year mode puts old literature above new (corr>0.6)", cy > 0.6, cy)
# n=68 -> the standard error of r is ~0.12, so anything under ~0.35 is indistinguishable
# from no relationship; the claim is that sim mode stops ordering by year, not that r==0.
check("sim mode drops the year ordering (r<0.35, and far below year mode)",
      abs(cs) < 0.35 and (cy - cs) > 0.45, (cy, cs))
ctx.eval("GRAPH.pos = graphLayout(GRAPH.data, GRAPH.mode);")

# --- neighbours / selection -------------------------------------------------
ctx.eval("GRAPH.sel = 'W2011301426'; var __nb = graphNeighbours();")
check("neighbours includes edge partner", ctx.eval("__nb['W3035965352']") == 1)
check("neighbours empty without selection",
      ctx.eval("GRAPH.sel=null, JSON.stringify(graphNeighbours())") == "{}")
check("graphNode finds by id", ctx.eval("graphNode('W2011301426').id") == "W2011301426")
check("graphNode null when missing", ctx.eval("graphNode('nope')") is None)

# --- render -----------------------------------------------------------------
ctx.eval("GRAPH.sel = null; renderGraph();")
svg = ctx.eval("$('gmap').innerHTML")
check("renderGraph emits svg", svg.startswith("<svg id=\"gsvg\"") and svg.endswith("</svg>"))
check("renderGraph draws one circle per node", svg.count("<circle") == nnode, svg.count("<circle"))
check("renderGraph draws one line per edge", svg.count("<line") == len(g["edges"]), svg.count("<line"))
check("renderGraph wires node ids", 'data-gid="W2011301426"' in svg)
check("renderGraph has zoom group", 'id="gzoom"' in svg)
ctx.eval("GRAPH.sel='W2011301426'; renderGraph();")
svg2 = ctx.eval("$('gmap').innerHTML")
check("selection marks node .sel", 'class="gnode sel"' in svg2)
check("selection dims non-neighbours", 'opacity="0.25"' in svg2)

ctx.eval("renderGraphSide();")
side = ctx.eval("$('gside').innerHTML")
for want in ["核心文献", "先行文献", "后续文献", "发表年份", "gyear", "glist"]:
    check("side panel has %s" % want, want in side)
check("side panel lists selected node detail", "gdetail" in side)
ctx.eval("GRAPH.sel=null; renderGraphSide();")
side = ctx.eval("$('gside').innerHTML")
check("side panel drops detail when deselected", "gdetail" not in side)

# --- exports ----------------------------------------------------------------
ctx.eval("GRAPH.sel=null; downloadGraphCSV();")
cap = ctx.eval("JSON.stringify(__cap)"); cap = json.loads(cap)
rows = cap["c"].lstrip("\ufeff").split("\n")
check("CSV row per node+edge+header", len(rows) == 1 + nnode + len(g["edges"]), len(rows))
check("CSV has BOM + csv mime", cap["c"][0] == "\ufeff" and "csv" in cap["m"])
check("CSV header names", rows[0].startswith("kind,id_or_source"), rows[0])
check("CSV node row is a node", rows[1].startswith('"node"'), rows[1][:40])
check("CSV edge row is an edge", rows[-1].startswith('"edge","W'), rows[-1][:50])
check("CSV quotes escaped", '"' in rows[1][1:] and '""' in rows[1])
ctx.eval("download = __grab; downloadBlob = __grab; __cap=null; downloadGraphSVG();")
cap = ctx.eval("JSON.stringify(__cap)"); cap = json.loads(cap)
body = (cap or {}).get("c", {}).get("parts", [""])[0]
check("SVG export is an XML svg document",
      cap is not None and body.startswith("<?xml") and "<svg" in body and "</svg>" in body)
check("SVG export resets zoom on the clone",
      ctx.eval("__lastClone.__sub.transform") == "translate(0,0) scale(1)",
      ctx.eval("__lastClone.__sub.transform"))
check("SVG export drops hover titles for print", ".gnode title{display:none}" in body)
check("SVG export names file after seed", "scholarmate-graph-W3035965352.svg" == cap["n"], cap["n"])

print()
print("FAILURES: %d" % len(fails))
for f in fails: print("  - " + f)
sys.exit(1 if fails else 0)
