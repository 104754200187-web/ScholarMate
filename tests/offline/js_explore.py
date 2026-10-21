"""Syntax-check and smoke-eval the inline frontend JS (no browser here).

esprima gives the syntax gate; quickjs runs the same code against a DOM stub so
the exploration module can actually be called (panel HTML, direction cards,
mode labels, path bookkeeping), not just parsed.
"""
import re
import sys

import esprima

import _paths  # noqa: F401
import web  # noqa: E402

PAGE = web.PAGE
scripts = re.findall(r"<script[^>]*>(.*?)</script>", PAGE, re.S)
assert scripts, "no <script> block found"
js = "\n;\n".join(scripts)
print(f"script blocks: {len(scripts)}, chars: {len(js)}")

esprima.parseScript(js)
print("esprima: syntax ok")

# ---- runtime harness ------------------------------------------------------
STUB = r"""
var LOG = [];
function El(id) {
  this.id = id; this.innerHTML = ""; this.textContent = ""; this.value = "";
  this.style = {}; this.classList = {add:function(){}, remove:function(){},
    toggle:function(){}, contains:function(){return false;}};
  this.children = []; this.disabled = false; this.open = false; this.checked = false;
  this.dataset = {}; this.hidden = false;
}
El.prototype.addEventListener = function(){};
El.prototype.appendChild = function(c){ this.children.push(c); return c; };
El.prototype.removeChild = function(){};
El.prototype.closest = function(){ return null; };
El.prototype.querySelector = function(){ return null; };
El.prototype.querySelectorAll = function(){ return []; };
El.prototype.scrollIntoView = function(){};
El.prototype.focus = function(){};
El.prototype.remove = function(){};
El.prototype.setAttribute = function(){};
El.prototype.getAttribute = function(){ return null; };
El.prototype.insertBefore = function(c){ return c; };
El.prototype.cloneNode = function(){ return new El(this.id); };

var _els = {};
function _get(id) { if (!_els[id]) _els[id] = new El(id); return _els[id]; }
var document = {
  getElementById: _get,
  createElement: function(t){ return new El("new-" + t); },
  querySelector: function(){ return _get("q"); },
  querySelectorAll: function(){ return []; },
  addEventListener: function(){},
  body: new El("body"), documentElement: new El("html"),
};
var window = {addEventListener:function(){}, matchMedia:function(){ return {matches:false, addEventListener:function(){}}; },
              getSelection:function(){ return {length:0, toString:function(){return "";}}; },
              print:function(){}, scrollTo:function(){}, location:{hash:""}};
var location = window.location;
var navigator = {clipboard:{writeText:function(){ return Promise.resolve(); }}, userAgent:"test"};
var localStorage = {getItem:function(){return null;}, setItem:function(){}, removeItem:function(){}};
var setTimeout = function(fn){ return 0; };
var setInterval = function(){ return 0; };
var clearTimeout = function(){};
var fetch = function(){ return Promise.resolve({ok:true, json:function(){ return Promise.resolve({}); },
                                                text:function(){ return Promise.resolve(""); },
                                                blob:function(){ return Promise.resolve({}); }}); };
var console = {log:function(){}, error:function(){}, warn:function(){}};
var alert = function(){};
var URL = {createObjectURL:function(){return "blob:x";}, revokeObjectURL:function(){}};
var Blob = function(){};
var FileReader = function(){};
"""
QUICK = STUB + "\n" + js + r"""
;__checks()
"""
checks = r"""
function __checks() {
  var out = [];
  function ok(name, cond) { out.push((cond ? "PASS " : "FAIL ") + name); }
  ok("modeLabel explore", modeLabel("explore") === "领域探索");
  ok("modeLabel survey", modeLabel("survey") === "综述报告");
  ok("modeLabel search", modeLabel("search") === "仅搜索论文");
  ok("exploreStart exists", typeof exploreStart === "function");
  ok("exploreDeepen exists", typeof exploreDeepen === "function");
  ok("exploreSurvey exists", typeof exploreSurvey === "function");
  ok("restoreExplore exists", typeof restoreExplore === "function");
  ok("exploreBack exists", typeof exploreBack === "function");
  var d = {name: "关系抽取", en: "relation extraction", why: "把实体连成网络",
           queries: ["biomedical relation extraction"],
           refs: [{title: "BioBERT", year: 2019, citations: 7499, url: "https://x", venue: "Bioinformatics"}]};
  var card = dirCardHTML(d, 0);
  ok("card has name", card.indexOf("关系抽取") >= 0);
  ok("card has english", card.indexOf("relation extraction") >= 0);
  ok("card has ref link", card.indexOf("BioBERT") >= 0 && card.indexOf("https://x") >= 0);
  ok("card has deepen button", card.indexOf("exploreDeepen(0)") >= 0);
  EXPLORE.topic = "生物医学文本挖掘";
  EXPLORE.last = [d];
  var panel = explorePanelHTML();
  ok("panel has topic", panel.indexOf("生物医学文本挖掘") >= 0);
  ok("panel has survey button", panel.indexOf("expsurvey") >= 0);
  EXPLORE.selected = {};
  exploreToggle(0);
  ok("toggle selects", EXPLORE.selected["关系抽取"] === true);
  exploreToggle(0);
  ok("toggle deselects", !EXPLORE.selected["关系抽取"]);
  EXPLORE.turns = [{action: "scope", topic: "T", parent: {}, path: [], directions: [d, {name: "B"}]},
                   {action: "deepen", topic: "T", parent: {name: "关系抽取"},
                    path: [], directions: [{name: "C"}]}];
  restoreExplore({id: "r1", question: "生物医学文本挖掘", explore: EXPLORE.turns});
  ok("restore keeps last layer", EXPLORE.last.length === 1 && EXPLORE.last[0].name === "C");
  ok("restore rebuilds path", EXPLORE.path.length === 1 && EXPLORE.path[0] === "关系抽取");
  ok("restore collects seen", EXPLORE.seen.join(",") === "关系抽取,B,C");
  ok("restore sets runId", EXPLORE.runId === "r1");
  exploreBack();
  ok("back pops a layer", EXPLORE.turns.length === 1 && EXPLORE.last.length === 2);

  // --- 追问对话：补检索后的气泡与工作集刷新
  var bubble = chatBubbleHTML({role: "assistant", content: "答案 [21]", counts: {ok: 2}, duration_s: 7.5,
                               fetched: [{query: "llm protein structure prediction", added: 3,
                                          titles: ["AlphaFold and language models", "ESM-2", "T"]}],
                               papers_added: 3, needs_new_search: false});
  ok("bubble shows the fetch", bubble.indexOf("本轮自动补检索") >= 0);
  ok("bubble shows the query", bubble.indexOf("protein structure prediction") >= 0);
  ok("bubble lists new titles", bubble.indexOf("ESM-2") >= 0);
  ok("bubble shows verification", bubble.indexOf("引用校验") >= 0);
  var still = chatBubbleHTML({role: "assistant", content: "没找到", counts: {ok: 1},
                              needs_new_search: true, search_query: "q", papers_added: 0});
  ok("unmet-need bubble offers a full search", still.indexOf("开一次完整检索") >= 0);
  var after = chatBubbleHTML({role: "assistant", content: "仍然不足", counts: {ok: 1},
                              needs_new_search: true, search_query: "q", papers_added: 2});
  ok("bubble distinguishes exhausted retries", after.indexOf("补检索后材料仍不足") >= 0);
  // --- 参考文献 / 工作论文集 折叠
  var md = "## 摘要\n\n结论见 [1]。\n\n## 参考文献\n\n[1] Real Paper (2020)\n[2] Second (2021)";
  var sp = splitRefs(md);
  ok("splitRefs cuts the body", sp.body.indexOf("参考文献") < 0 && sp.body.indexOf("结论见") >= 0);
  ok("splitRefs keeps the entries", sp.refs.indexOf("[1] Real Paper") >= 0);
  var out1 = render(md, []);
  ok("references fold into a details", out1.indexOf('details class="refs"') >= 0);
  ok("references count shown", out1.indexOf("2 条") >= 0);
  ok("body citation survives", out1.indexOf('href="#cite-1"') >= 0);
  ok("folded refs are not in the body text", out1.indexOf("<details") > out1.indexOf('href="#cite-1"'));
  var out2 = render("## 摘要\n\n见 [1]。", []);
  ok("no refs section => no details", out2.indexOf("details") < 0);
  var out3 = render("一句话 [2]。\n\nReferences:\n[2] Something (2019)", []);
  ok("english heading folds too", out3.indexOf('details class="refs"') >= 0);
  ok("english refs not left inline", out3.indexOf("Something (2019)") > out3.indexOf("<details"));
  ok("revealPaper exists", typeof revealPaper === "function");
  revealPaper("cite-3");
  return out.join("\n");
}
"""
import quickjs  # noqa: E402

ctx = quickjs.Context()
ctx.eval(STUB)
ctx.eval(js)
ctx.eval(checks)
res = ctx.eval("__checks()")
print(res)
fails = [l for l in res.split("\n") if l.startswith("FAIL")]
print(f"\n{len(res.splitlines()) - len(fails)}/{len(res.splitlines())} checks passed")
sys.exit(1 if fails else 0)
