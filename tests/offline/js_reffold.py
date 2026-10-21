"""Run the page's own splitRefs/render over every real stored answer."""
import glob, json, re, sys
import esprima, quickjs
from _paths import FIXTURES
import web

js = "\n;\n".join(re.findall(r"<script[^>]*>(.*?)</script>", web.PAGE, re.S))
esprima.parseScript(js)

STUB = """
function El(id){this.id=id;this.innerHTML="";this.textContent="";this.value="";this.style={};
 this.classList={add:function(){},remove:function(){},toggle:function(){},contains:function(){return false;}};
 this.children=[];this.disabled=false;this.open=false;this.checked=false;this.dataset={};this.hidden=false;}
El.prototype.addEventListener=function(){};El.prototype.appendChild=function(c){return c;};
El.prototype.closest=function(){return null;};El.prototype.querySelector=function(){return null;};
El.prototype.querySelectorAll=function(){return [];};El.prototype.scrollIntoView=function(){};
El.prototype.focus=function(){};El.prototype.remove=function(){};El.prototype.setAttribute=function(){};
El.prototype.getAttribute=function(){return null;};
var _els={};function _get(id){if(!_els[id])_els[id]=new El(id);return _els[id];}
var document={getElementById:_get,createElement:function(t){return new El(t);},
 querySelector:function(){return _get("q");},querySelectorAll:function(){return [];},
 addEventListener:function(){},body:new El("body"),documentElement:new El("html")};
var window={addEventListener:function(){},matchMedia:function(){return {matches:false,addEventListener:function(){}};},
 getSelection:function(){return {length:0,toString:function(){return "";}};},print:function(){},scrollTo:function(){},location:{hash:""}};
var location=window.location;var navigator={clipboard:{writeText:function(){return Promise.resolve();}},userAgent:"t"};
var localStorage={getItem:function(){return null;},setItem:function(){},removeItem:function(){}};
var setTimeout=function(){return 0;};var setInterval=function(){return 0;};var clearTimeout=function(){};
var fetch=function(){return Promise.resolve({ok:true,json:function(){return Promise.resolve({});}});};
var console={log:function(){},error:function(){},warn:function(){}};var alert=function(){};
var URL={createObjectURL:function(){return "blob:x";},revokeObjectURL:function(){}};
var Blob=function(){};var FileReader=function(){};
"""
ctx = quickjs.Context()
ctx.eval(STUB); ctx.eval(js)

answers = []
for f in sorted(glob.glob(str(FIXTURES / "history" / "*.json"))):
    try: d = json.load(open(f, encoding="utf-8"))
    except Exception: continue
    a = d.get("answer") or ""
    if a and "[1]" in a: answers.append((d.get("mode"), a))
print(f"real answers checked: {len(answers)}")

fails = []
folded = 0
for mode, a in answers:
    if not ("参考文献" in a or re.search(r"^[#>*_\s]*References", a, re.M)):
        continue
    ctx.set("__md", a)
    sp = ctx.eval("JSON.stringify(splitRefs(__md))")
    d = json.loads(sp)
    out = ctx.eval("render(__md, [])")
    ctx.eval("var __out = render(__md, [])")
    out = ctx.eval("__out")
    has_heading_in_body = bool(re.search(r"^[#>*_\s]*(参考文献|References)[\s*_:：]*", d["body"], re.M))
    ok_fold = 'details class="refs"' in out
    ok_entries = len(re.findall(r"\[\d+\]", d["refs"])) >= 3
    ok_body_cites = "[1]" in out.split('<details class="refs"')[0] or "[2]" in out.split('<details class="refs"')[0]
    folded += 1
    if not (ok_fold and ok_entries and not has_heading_in_body):
        fails.append((mode, ok_fold, ok_entries, has_heading_in_body, d["refs"][:80]))

print(f"answers with a reference list: {folded}")
for f in fails:
    print("FAIL", f[0], "folded:", f[1], "entries:", f[2], "heading leaked into body:", f[3], "|", f[4])
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
