"""Card button -> graph panel -> /citegraph -> rendered svg, plus 返回结果."""
import json, re, sys
import esprima, quickjs
from _paths import FIXTURES
import web

js = "\n;\n".join(re.findall(r"<script[^>]*>(.*?)</script>", web.PAGE, re.S))
esprima.parseScript(js)

STUB = """
function El(id){this.id=id;this.innerHTML="";this.textContent="";this.value="";this.style={};
 this.classList={_s:{},add:function(c){this._s[c]=1;},remove:function(c){delete this._s[c];},
   toggle:function(c){},contains:function(c){return !!this._s[c];}};
 this.children=[];this.disabled=false;this.open=false;this.checked=false;this.dataset={};this.hidden=false;}
El.prototype.addEventListener=function(){};El.prototype.appendChild=function(c){return c;};
El.prototype.closest=function(){return null;};
El.prototype.querySelector=function(s){return s.charAt(0)==="#"?_get(s.slice(1)):null;};
El.prototype.querySelectorAll=function(){return [];};El.prototype.scrollIntoView=function(){};
El.prototype.focus=function(){};El.prototype.remove=function(){};
El.prototype.setAttribute=function(k,v){this[k]=v;};El.prototype.getAttribute=function(k){return this[k]||null;};
El.prototype.cloneNode=function(){var e=new El(this.id);e.innerHTML=this.innerHTML;return e;};
El.prototype.getBoundingClientRect=function(){return {left:0,top:0,width:960,height:640};};
var _els={};function _get(id){if(!id)return null;if(!_els[id])_els[id]=new El(id);return _els[id];}
var document={getElementById:_get,createElement:function(t){return new El(t);},
 querySelector:function(){return _get("q");},querySelectorAll:function(){return [];},
 __clicks:[],addEventListener:function(t,fn){ if(t==="click") document.__clicks.push(fn); },
 body:new El("body"),documentElement:new El("html")};
var window={addEventListener:function(){},matchMedia:function(){return {matches:false,addEventListener:function(){}};},
 getSelection:function(){return {length:0,toString:function(){return "";}};},print:function(){},scrollTo:function(){},location:{hash:""}};
var location=window.location;var navigator={clipboard:{writeText:function(){return Promise.resolve();}},userAgent:"t"};
var localStorage={getItem:function(){return null;},setItem:function(){},removeItem:function(){}};
var setTimeout=function(){return 0;};var setInterval=function(){return 0;};var clearTimeout=function(){};
var requestAnimationFrame=function(){return 0;};
var __fetches=[];var __payload=null;
var fetch=function(url,opt){ __fetches.push({url:url, body:(opt||{}).body||""});
  return Promise.resolve({ok:true, json:function(){ return Promise.resolve(JSON.parse(__payload)); }});};
var console={log:function(){},error:function(){},warn:function(){}};var alert=function(){};
var URL={createObjectURL:function(){return "blob:x";},revokeObjectURL:function(){}};
var Blob=function(p){this.parts=p||[];};var FileReader=function(){};
var __cap=null;function __grab(n,c,m){__cap={n:n,c:c,m:m};}
"""

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

g = json.load(open(FIXTURES / "graph_af2.json", encoding="utf-8"))
if "graph" in g: g = g["graph"]
ctx.set("__payload", json.dumps({"ok": True, "graph": g}))
ctx.eval("__payload = __payload;")

# 1. the card itself carries the button --------------------------------------
card = ctx.eval('paperItemHTML({title:"Array programming with NumPy", source:"s2",'
                ' doi:"10.1038/s41586-020-2649-2", year:2020, citations:23200}, 0)')
check("search/ask card has the graph button", "data-gseed=" in card and "关系图谱" in card)
check("card button carries the DOI", 'data-gseed="10.1038/s41586-020-2649-2"' in card)
noc = ctx.eval('paperItemHTML({title:"", source:"s2", year:2020, citations:1}, 0)')
check("card without any handle has no button", "data-gseed=" not in noc)

# 2. delegated click on that button opens the panel and starts the fetch ------
ctx.eval('document.__clicks.forEach(function(h){ h({target:{closest:function(s){'
         'return s===".gbtn"?{getAttribute:function(){return "10.1038/s41586-020-2649-2";}}:null;}},'
         ' preventDefault:function(){}}); });')
panel = ctx.eval('$("answer").innerHTML')
check("click opens the graph panel", 'id="gpanel"' in panel and 'id="gseed"' in panel, panel[:150])
check("canvas is mounted only once data arrives", 'id="gmap"' not in panel)
check("panel seed box is prefilled", 'value="10.1038/s41586-020-2649-2"' in panel)
check("panel asks for the network right away", "正在抓取引用网络" in ctx.eval('$("gstatus").innerHTML'))
ctx.eval("""
function __cg() { return __fetches.filter(function (f) { return f.url === "/citegraph"; }); }
""")
fetches = json.loads(ctx.eval("JSON.stringify(__cg())"))
check("click posts exactly one /citegraph", len(fetches) == 1, json.loads(ctx.eval("JSON.stringify(__fetches)")))
check("post body carries the seed",
      json.loads(fetches[0]["body"] or "{}").get("seed") == "10.1038/s41586-020-2649-2", fetches[0]["body"])

# 3. draining the promise renders the whole graph -----------------------------
drain()
check("promise chain settled", ctx.eval("typeof GRAPH.data") == "object" and ctx.eval("GRAPH.data") is not None)
status = ctx.eval('$("gstatus").innerHTML')
check("status summarises the graph", ("68 篇文献" in status) and ("条关系" in status), status)
check("off-field filtering is disclosed", "跨领域" in status, status)
svg = ctx.eval('$("gmap").innerHTML')
check("svg drew every node", svg.count("<circle") == len(g["nodes"]), svg.count("<circle"))
check("svg drew every edge", svg.count("<line") == len(g["edges"]), svg.count("<line"))
sides = ctx.eval('$("gside").innerHTML')
check("side panel rendered core list", "核心文献" in sides and "先行文献" in sides)
check("mode buttons exist", 'id="gmode-year"' in ctx.eval('$("gbody").innerHTML')
      and 'id="gmode-sim"' in ctx.eval('$("gbody").innerHTML'))

# 4. re-sorting does not refetch; rebuilding does -----------------------------
before = len(json.loads(ctx.eval("JSON.stringify(__cg())")))
ctx.eval("GRAPH.mode='sim'; relayoutGraph();")
after = len(json.loads(ctx.eval("JSON.stringify(__cg())")))
check("re-sorting is client-side only (no new request)", after == before, (before, after))
ctx.eval('$("gseed").value = "10.1103/PhysRevLett.116.061102"; graphRebuild();')
fetches = json.loads(ctx.eval("JSON.stringify(__cg())"))
check("换一篇 refetches with the new seed", len(fetches) == 2 and
      json.loads(fetches[1]["body"]).get("seed") == "10.1103/PhysRevLett.116.061102", fetches[-1])
ctx.eval('$("gseed").value = "   "; graphRebuild();')
check("empty seed is refused client-side",
      len(json.loads(ctx.eval("JSON.stringify(__cg())"))) == 2 and "先填" in ctx.eval('$("gstatus").innerHTML'))

# 5. ← 返回结果 restores whatever the reader was looking at -------------------
ctx.eval('current = {mode:"search", question:"numpy", papers:[' + card.replace("\n", " ") + 'and null][0]'
         if False else 'var __p = {title:"Array programming with NumPy", source:"s2", doi:"10.1038/s41586-020-2649-2", year:2020, citations:23200};\n'
         'current = {mode:"search", question:"numpy", papers:[__p], sort_label:"相关性", scores:null};')
ctx.eval("graphBack();")
back = ctx.eval('$("answer").innerHTML')
check("← 返回结果 restores the result list", "paper" in back and "关系图谱" in back, back[:120])
check("graph panel is gone after going back", 'id="gpanel"' not in back)
ctx.eval("current = null; graphBack();")
check("空闲时返回回到首屏", ctx.eval('$("answer").innerHTML') == "")

print()
print("FAILURES: %d" % len(fails))
for f in fails: print("  - " + f)
sys.exit(1 if fails else 0)
