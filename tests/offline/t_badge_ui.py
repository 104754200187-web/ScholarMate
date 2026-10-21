"""The 无法核对 badge in the page: chips, claim rows, and old snapshots."""
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
var fetch=function(url,opt){ return Promise.resolve({ok:true, json:function(){return Promise.resolve({nodes:[],edges:[]});}});};
var console={log:function(){},error:function(){},warn:function(){}};var alert=function(){};
var URL={createObjectURL:function(){return "blob:x";},revokeObjectURL:function(){}};
var Blob=function(p){this.parts=p||[];};var FileReader=function(){};
function __grab(n,c,m){};var __cap=null;
"""

ctx = quickjs.Context()
ctx.eval(STUB)
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

SNAP = FIXTURES / "history" / "20261015_154020_081495.json"
snap = json.load(open(SNAP, encoding="utf-8"))
payload = dict(snap)
payload["counts"] = {"claims": web._count_claims(snap["audits"]["claims"])}
ctx.set("__p", json.dumps(payload, ensure_ascii=False))
ctx.eval("var __d = JSON.parse(__p); window.EX = __d.exports || {}; window.SNAP = JSON.stringify(__d);"
         "current = __d; renderAudits(__d);")
drain()

# ---- 1. the saved verdict renders as unchecked, not as a failure ------------
raw4 = [c for c in snap["audits"]["claims"]["verdicts"] if c["id"] == 4][0]
ctx.set("__c", json.dumps(raw4, ensure_ascii=False))
check("the page re-derives the old verdict as unfetched",
      ctx.eval("badgeFor(JSON.parse(__c)).key") == "unfetched", ctx.eval("badgeFor(JSON.parse(__c)).key"))
check("its badge is the 无法核对 one",
      "无法核对" in ctx.eval("badgeFor(JSON.parse(__c)).label"))
check("the reason is on the badge",
      "download failed" in ctx.eval("badgeFor(JSON.parse(__c)).note"),
      ctx.eval("badgeFor(JSON.parse(__c)).note"))
check("a claim with a plain abstract verdict is still 不支持",
      ctx.eval('badgeFor({supported:false, supporting:[], evidence:[{n:1, source:"abstract", note:"abstract only"}]}).key') == "unsupported")
check("a new-format unfetched verdict is taken as stored",
      ctx.eval('badgeFor({supported:false, badge:"unfetched", badge_label:"无法核对（原文未取到）", badge_note:"没取到"}).label') == "无法核对（原文未取到）")

chips = ctx.eval('$("chips").innerHTML')
claimchip = re.search(r'<span class="chip [^"]*"[^>]*>逐句证据[^<]*</span>', chips).group(0)
visible = re.sub(r'title="[^"]*"', "", claimchip)
check("the chip counts it as unchecked, not as a failure",
      "无法核对 1" in visible and "不支持" not in visible, claimchip)
check("the chip is a warning, not an error",
      'class="chip warn"' in claimchip, claimchip)
check("the chip explains itself on hover", "不等于不支持" in claimchip, claimchip)
check("the chip's denominator is the claim count", "逐句证据 7/8 支持" in chips, chips[:300])

html = ctx.eval('$("answer").innerHTML')
check("the claim row no longer says ❌ for that claim",
      "❌" not in html or "通过判别器辅助训练" not in html, html[-400:])
rows = [h for h in html.split('<div class="claim ') if h.startswith("unf")]
check("the unchecked claim gets its own row style", len(rows) == 1, len(rows))
check("the row warns instead of failing", len(rows) == 1 and rows[0].startswith("unf") and "\u26a0" in rows[0], rows[:1])
check("no claim is rendered as unsupported any more", 'class="claim unsup"' not in html,
      [h[:80] for h in html.split('<div class="claim ') if h.startswith("unsup")])

# ---- 2. a new-format run: the row says which section the claim came from ----
fresh = dict(payload)
fresh["audits"] = json.loads(json.dumps(payload["audits"]))
fresh["audits"]["claims"]["verdicts"] = [dict(raw4, badge="unfetched",
                                              badge_label="无法核对（原文未取到）",
                                              badge_note="本次未取到可核对原文（text extraction timed out after 20s）",
                                              section="跨细胞类型泛化预测（cross-cell-type perturbation generalization）")]
fresh["counts"] = {"claims": web._count_claims(fresh["audits"]["claims"])}
ctx.set("__p2", json.dumps(fresh, ensure_ascii=False))
ctx.eval("var __e=JSON.parse(__p2); window.EX=__e.exports||{}; window.SNAP=JSON.stringify(__e); current=__e; renderAudits(__e);")
drain()
h2 = ctx.eval('$("answer").innerHTML')
row2 = [h for h in h2.split('<div class="claim ') if h.startswith("unf")]
check("a new-format row keeps the stored reason",
      row2 and "text extraction timed out after 20s" in row2[0], row2[:1])
check("and names the section the claim came from",
      row2 and "声明 4（跨细胞类型泛化预测" in row2[0], row2[:1])
check("the chip on a fresh run still counts one unchecked claim",
      "无法核对 1" in ctx.eval('$("chips").innerHTML'), ctx.eval('$("chips").innerHTML')[:200])

# ---- 3. external links only become http(s) hrefs ---------------------------
check("javascript: links are refused",
      ctx.eval('safeHref("javascript:alert(1)")') == "")
check("data: links are refused",
      ctx.eval('safeHref("data:text/html,<script>x</script>")') == "")
check("file: links are refused", ctx.eval('safeHref("file:///etc/passwd")') == "")
check("normal https links survive",
      ctx.eval('safeHref("https://example.com/a.pdf")') == "https://example.com/a.pdf")
check("a quote cannot break out of an href attribute",
      '" onmouseover="' not in ctx.eval('paperLinks({pdf_url:"https://e.com/\\" onmouseover=\\"x", url:""}, 0)'))
check("an unsafe PDF URL is not rendered as a link",
      "javascript:" not in ctx.eval('paperLinks({pdf_url:"javascript:alert(1)", url:""}, 0)'))

print()
print("FAILED: " + ", ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
