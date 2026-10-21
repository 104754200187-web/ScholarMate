"""Per-browser LLM settings: page wiring, header payload and server refusal."""

import base64
import inspect
import json
import os
import re
import sys

import esprima
import quickjs

import _paths  # noqa: F401
import llm
import web

fails = []


def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name
          + ("" if cond else "  <- " + repr(extra)))
    if not cond:
        fails.append(name)


def b64(obj):
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


CFG = {"provider": "qwen", "api_key": "user-qwen-key",
       "model": "qwen-max", "base_url": "", "api_style": "chat"}
NEW_PROVIDERS = {
    "baidu": ("ERNIE-5.1", "cloud.baidu.com/doc/qianfan"),
    "hunyuan": ("hunyuan-turbos-latest", "cloud.tencent.com"),
    "minimax": ("MiniMax-M3.1-Flash", "minimaxi.com"),
    "groq": ("openai/gpt-oss-120b", "groq.com"),
    "openrouter": ("~openai/gpt-sol-latest", "openrouter.ai"),
    "cohere": ("command-a-plus-05-2026", "cohere.com"),
    "siliconflow": ("deepseek-ai/DeepSeek-V4-Flash", "siliconflow.cn"),
}

# ---- 1. the panel and its storage/header contract -------------------------
check("the page has a model settings panel", 'id="llmpanel"' in web.PAGE)
check("the settings dock is fixed in the top-left and collapsed by default",
      '#llmpanel { position:fixed; top:16px; left:20px;' in web.PAGE
      and '<details class="panel" id="llmpanel">' in web.PAGE)
check("the page has a blocking dialog for missing credentials",
      'id="llmmodal" hidden role="dialog" aria-modal="true"' in web.PAGE
      and 'id="llmmodalgo"' in web.PAGE
      and 'id="llmmodalclose"' in web.PAGE)
check("it offers DeepSeek, OpenAI, Qwen and custom",
      all(f'value="{p}"' in web.PAGE for p in ("deepseek", "openai", "qwen", "custom")))
check("it offers the other major hosted LLM providers",
      all(f'value="{p}"' in web.PAGE for p in
          ("anthropic", "gemini", "xai", "mistral", "moonshot", "zhipu", "doubao",
           *NEW_PROVIDERS)))
check("the settings panel links to each provider's official docs",
      'id="llmdoclink"' in web.PAGE and "官方调用文档" in web.PAGE)
check("the API key field is masked", 'id="llmkey" type="password"' in web.PAGE)
check("the page says the key stays in this browser", "localStorage" in web.PAGE)
check("the page sends the config in a dedicated header",
      "X-LLM-Config" in web.PAGE and "llmHeaders" in web.PAGE)
check("the page marks free tiers and trial credits",
      "✅ 免费层" in web.PAGE and "🎁 新用户赠送/试用" in web.PAGE
      and "freeModels:" in web.PAGE and "trialModels:" in web.PAGE)
check("the missing-credentials dialog offers free access",
      all(f'id="{el}"' in web.PAGE for el in
          ("llmfreeselect", "llmfreeuse", "llmfreestatus", "llmfreetag")))
check("the server exposes only configured free options",
      'elif self.path == "/free-llm":' in inspect.getsource(web.Handler.do_GET)
      and "llm.free_options()" in inspect.getsource(web.Handler.do_GET))
endpoint_headers = (
    'fetch("/ask", {method:"POST", headers:llmHeaders',
    'fetch("/chat", {method: "POST", headers: llmHeaders',
    'fetch("/resume", {method: "POST", headers: llmHeaders',
    'fetch("/explore", {method: "POST", headers: llmHeaders',
    'fetch("/rerank", {method:"POST", headers:llmHeaders',
)
check("ask/chat/resume/explore/rerank all use the header",
      all(snippet in web.PAGE for snippet in endpoint_headers),
      [snippet for snippet in endpoint_headers if snippet not in web.PAGE])

# ---- 2. decode one header without touching any persistent store -----------
decoded = web._decode_llm_config(b64(CFG))
check("a valid header decodes", decoded and decoded["provider"] == "qwen", decoded)
check("decoded config keeps the user key",
      decoded["api_key"] == "user-qwen-key", decoded)
for raw in ("!!!not-base64!!!", b64([]), b64({"provider": "qwen"}),
            b64({"provider": "custom", "api_key": "k", "model": "m",
                 "base_url": "http://127.0.0.1/v1"})):
    try:
        web._decode_llm_config(raw)
        check("invalid header rejected: " + raw[:24], False, "accepted")
    except ValueError:
        check("invalid header rejected: " + raw[:24], True)


class FakeHandler(web.Handler):
    def __init__(self, headers, owner=None):
        self.headers = headers
        self.sent = []
        if owner is not None:
            self._owner = owner

    def _send(self, code, ctype, body, headers=None):
        self.sent.append((code, body))


real_allow = llm.server_default_allowed
try:
    llm.server_default_allowed = lambda: False
    request = FakeHandler({})
    result = request._llm_for_request()
    check("without a user config the request is refused",
          result is web._LLM_REJECTED, result)
    check("the refusal tells the user where to configure",
          request.sent and request.sent[0][0] == 400
          and "模型与 API Key" in request.sent[0][1].decode("utf-8"),
          request.sent)

    request = FakeHandler({web.LLM_CONFIG_HEADER: b64(CFG)})
    result = request._llm_for_request()
    check("a valid request header is accepted",
          result and result["model"] == "qwen-max", result)

    llm.server_default_allowed = lambda: True
    request = FakeHandler({})
    check("server fallback is possible only when explicitly enabled",
          request._llm_for_request() is None)
finally:
    llm.server_default_allowed = real_allow

# ---- 2b. server-owned free access and its quota ---------------------------
free_env = ("GEMINI_API_KEY", "GROQ_API_KEY", "ZHIPU_API_KEY",
            "OPENROUTER_API_KEY", "COHERE_API_KEY")
saved_free_env = {key: os.environ.get(key) for key in free_env}
real_load_dotenv = llm._load_dotenv
real_free_daily = web.FREE_LLM_DAILY_REQUESTS
real_free_global = web.FREE_LLM_GLOBAL_DAILY_REQUESTS
saved_usage = dict(web._free_llm_usage)
saved_global = dict(web._free_llm_global)
try:
    llm._load_dotenv = lambda path=".env": None
    for key in free_env:
        os.environ.pop(key, None)
    web.FREE_LLM_DAILY_REQUESTS = 2
    web.FREE_LLM_GLOBAL_DAILY_REQUESTS = 10
    web._free_llm_usage.clear()
    web._free_llm_global.update({"day": "", "used": 0})

    request = FakeHandler({web.FREE_LLM_HEADER: b64({
        "provider": "gemini", "model": "gemini-3.8-flash"})})
    result = request._llm_for_request()
    check("a free request without a server key is refused",
          result is web._LLM_REJECTED and request.sent[0][0] == 503, request.sent)

    os.environ["GEMINI_API_KEY"] = "server-free-key"
    owner = "a" * 32
    request = FakeHandler({web.FREE_LLM_HEADER: b64({
        "provider": "gemini", "model": "gemini-3.8-flash"})}, owner=owner)
    result = request._llm_for_request()
    check("a configured free request is accepted",
          result and result["free_access"] and result["api_key"] == "server-free-key",
          result)
    check("the free choice cannot inject a paid model",
          FakeHandler({web.FREE_LLM_HEADER: b64({
              "provider": "gemini", "model": "gemini-4-argon"})},
              owner=owner)._llm_for_request() is web._LLM_REJECTED)

    web.FREE_LLM_GLOBAL_DAILY_REQUESTS = 1
    request = FakeHandler({web.FREE_LLM_HEADER: b64({
        "provider": "gemini", "model": "gemini-3.8-flash"})}, owner="b" * 32)
    check("the all-site daily cap stops another browser",
          request._llm_for_request() is web._LLM_REJECTED
          and request.sent[0][0] == 429, request.sent)

    web.FREE_LLM_GLOBAL_DAILY_REQUESTS = 10
    web.FREE_LLM_DAILY_REQUESTS = 1
    request = FakeHandler({web.FREE_LLM_HEADER: b64({
        "provider": "gemini", "model": "gemini-3.8-flash"})}, owner=owner)
    check("the per-browser daily cap rejects the next request",
          request._llm_for_request() is web._LLM_REJECTED
          and request.sent[0][0] == 429, request.sent)

    web.FREE_LLM_DAILY_REQUESTS = 0
    request = FakeHandler({web.FREE_LLM_HEADER: b64({
        "provider": "gemini", "model": "gemini-3.8-flash"})}, owner="c" * 32)
    check("setting the free daily limit to zero disables the channel",
          request._llm_for_request() is web._LLM_REJECTED
          and request.sent[0][0] == 429, request.sent)
finally:
    llm._load_dotenv = real_load_dotenv
    web.FREE_LLM_DAILY_REQUESTS = real_free_daily
    web.FREE_LLM_GLOBAL_DAILY_REQUESTS = real_free_global
    web._free_llm_usage.clear()
    web._free_llm_usage.update(saved_usage)
    web._free_llm_global.clear()
    web._free_llm_global.update(saved_global)
    for key, value in saved_free_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

# ---- 3. task-scoped credentials, never in the persisted job ---------------
job_id = "20261021_120000_123456"
web._remember_job_llm(job_id, decoded)
check("a queued job can take its own config", web._take_job_llm(job_id) == decoded)
check("taking it removes the key from the pending map",
      web._take_job_llm(job_id) is None)

real_save = web.jobstore.save
saved_jobs = []
try:
    web.jobstore.save = lambda job: saved_jobs.append(json.loads(json.dumps(job)))
    free_job = {"id": "20261021_120000_654321", "status": "queued", "mode": "ask"}
    web._enqueue(free_job, "a" * 32, {
        "provider": "gemini", "model": "gemini-3.8-flash",
        "api_key": "must-not-persist", "free_access": True,
    })
    persisted = saved_jobs[0]
    check("a free job persists only its provider and model",
          persisted.get("llm_free_config") == {
              "provider": "gemini", "model": "gemini-3.8-flash"}
          and "api_key" not in json.dumps(persisted), persisted)
    check("a free job carries a visible free label",
          str(persisted.get("llm_label") or "").startswith("免费体验"), persisted)
    web._forget_job_llm(free_job["id"])
finally:
    web.jobstore.save = real_save


def page_js(initial=None):
    js = "\n;\n".join(re.findall(r"<script[^>]*>(.*?)</script>", web.PAGE, re.S))
    esprima.parseScript(js)
    stub = r'''
function El(id){this.id=id;this.innerHTML="";this.textContent="";this.value="";this.style={};
 this.classList={add:function(){},remove:function(){},toggle:function(){},contains:function(){return false;}};
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
 addEventListener:function(){},body:new El("body"),documentElement:new El("html")};
var window={addEventListener:function(){},matchMedia:function(){return {matches:false,addEventListener:function(){}};},
 getSelection:function(){return {length:0,toString:function(){return "";}};},print:function(){},scrollTo:function(){},location:{hash:""}};
var location=window.location;var navigator={clipboard:{writeText:function(){return Promise.resolve();}},userAgent:"t"};
var __store={};var localStorage={getItem:function(k){return (k in __store)?__store[k]:((k==="sm_llm_config_v1")?__initial:null);},
 setItem:function(k,v){__store[k]=v;},removeItem:function(k){delete __store[k];}};
var btoa=function(s){return s;};var TextEncoder=undefined;
var setTimeout=function(){return 0;};var setInterval=function(){return 0;};var clearTimeout=function(){};
var requestAnimationFrame=function(){return 0;};
var fetch=function(){return Promise.resolve({ok:true,json:function(){return Promise.resolve({});}});};
var console={log:function(){},error:function(){},warn:function(){}};var alert=function(){};
var URL={createObjectURL:function(){return "blob:x";},revokeObjectURL:function(){}};
var Blob=function(p){this.parts=p||[];};var FileReader=function(){};
'''
    ctx = quickjs.Context()
    ctx.set("__initial", json.dumps(initial) if initial else "null")
    ctx.eval(stub)
    ctx.eval(js)
    return ctx


ctx = page_js(CFG)
check("saved settings are restored on load",
      json.loads(ctx.eval("JSON.stringify(LLMCFG)"))["provider"] == "qwen")
summary = ctx.eval('$("llmsummary").textContent')
check("the summary names the model but never the key",
      "qwen-max" in summary and "user-qwen-key" not in summary, summary)
headers = json.loads(ctx.eval("JSON.stringify(llmHeaders({}))"))
check("the request header carries the config",
      json.loads(headers["X-LLM-Config"])["api_key"] == "user-qwen-key", headers)
check("llmRequire passes when a config exists", ctx.eval("llmRequire()") is True)
check("configured users do not see the missing-credentials dialog",
      ctx.eval('$("llmmodal").hidden') is True)

ctx.eval('$("llmprovider").value="anthropic"; llmProviderChanged(true);')
check("switching to Claude fills the documented default model",
      ctx.eval('$("llmmodel").value') == "claude-opus-5-5",
      ctx.eval('$("llmmodel").value'))
check("switching provider updates the official documentation link",
      "claude" in ctx.eval('$("llmdoclink").href'),
      ctx.eval('$("llmdoclink").href'))
ctx.eval('$("llmprovider").value="gemini"; llmProviderChanged(true);')
check("switching to Gemini fills its documented default model",
      ctx.eval('$("llmmodel").value') == "gemini-3.8-flash",
      ctx.eval('$("llmmodel").value'))
for provider, (model, docs_host) in NEW_PROVIDERS.items():
    ctx.eval(f'$("llmprovider").value="{provider}"; llmProviderChanged(true);')
    check(f"switching to {provider} fills its documented default model",
          ctx.eval('$("llmmodel").value') == model,
          ctx.eval('$("llmmodel").value'))
    check(f"switching to {provider} updates the official docs link",
          docs_host in ctx.eval('$("llmdoclink").href'),
          ctx.eval('$("llmdoclink").href'))

ctx.eval('$("llmprovider").value="openai"; llmProviderChanged(true);')
ctx.eval('$("llmkey").value="new-openai-key"; $("llmmodel").value="gpt-test"; llmSave();')
saved = json.loads(ctx.eval("JSON.stringify(LLMCFG)"))
check("saving the form switches provider and model",
      saved["provider"] == "openai" and saved["model"] == "gpt-test", saved)
check("the new key is stored in the browser", saved["api_key"] == "new-openai-key")
ctx.eval("llmClear();")
check("clearing removes the browser copy", ctx.eval("LLMCFG") is None)
check("clearing updates the summary",
      "未设置" in ctx.eval('$("llmsummary").textContent'))
check("clearing does not pop the dialog", ctx.eval('$("llmmodal").hidden') is True)

free = page_js(None)
free.eval('LLM_FREE_OPTIONS=[{provider:"zhipu",model:"GLM-4.7-Flash",'
          'label:"智谱 GLM",note:"智谱免费文本模型"}];'
          'LLM_FREE_LIMIT=5; LLM_FREE_GLOBAL_LIMIT=50;'
          'llmRenderFreeOptions();')
check("the free picker states both limits",
      "5" in free.eval('$("llmfreestatus").textContent')
      and "50" in free.eval('$("llmfreestatus").textContent'),
      free.eval('$("llmfreestatus").textContent'))
free.eval('LLM_FREE_ENABLED=false; llmRenderFreeOptions();')
check("a disabled free channel says so",
      "已关闭" in free.eval('$("llmfreestatus").textContent'),
      free.eval('$("llmfreestatus").textContent'))
free.eval('LLM_FREE_ENABLED=true; llmRenderFreeOptions();')
free.eval('$("llmfreeselect").value="zhipu"; llmFreeUse();')
check("choosing free access records only provider and model",
      json.loads(free.eval("JSON.stringify(LLM_FREE)")) == {
          "provider": "zhipu", "model": "GLM-4.7-Flash", "label": "智谱 GLM"},
      free.eval("JSON.stringify(LLM_FREE)"))
check("free access satisfies the submit guard",
      free.eval("llmRequire()") is True and free.eval('$("llmmodal").hidden') is True)
free_headers = json.loads(free.eval("JSON.stringify(llmHeaders({}))"))
check("free access uses its own header and never a user key",
      "X-LLM-Free-Config" in free_headers
      and "X-LLM-Config" not in free_headers
      and "api_key" not in json.loads(free_headers["X-LLM-Free-Config"]),
      free_headers)
check("the summary says this run uses free access",
      "免费体验" in free.eval('$("llmsummary").textContent'),
      free.eval('$("llmsummary").textContent'))
free.eval('$("llmprovider").value="gemini";'
          '$("llmmodel").value="gemini-3.8-flash"; llmUpdateFreeTag();')
check("a free model is labelled in the form",
      "免费层" in free.eval('$("llmfreetag").textContent'),
      free.eval('$("llmfreetag").textContent'))
free.eval('$("llmprovider").value="zhipu"; $("llmmodel").value="glm-5.3";'
          '$("llmkey").value="own-key"; llmSave();')
check("saving the user's own key clears the free selection",
      free.eval("LLM_FREE") is None
      and json.loads(free.eval("JSON.stringify(LLMCFG)"))["api_key"] == "own-key")
own_headers = json.loads(free.eval("JSON.stringify(llmHeaders({}))"))
check("a user key takes precedence over free access after saving",
      "X-LLM-Config" in own_headers and "X-LLM-Free-Config" not in own_headers,
      own_headers)

empty = page_js(None)
check("a fresh browser has no credentials", empty.eval("LLMCFG") is None)
check("a fresh browser keeps the settings dock folded",
      empty.eval('$("llmpanel").open') is False)
check("a fresh browser keeps the dialog closed until a question is submitted",
      empty.eval('$("llmmodal").hidden') is True)
check("a fresh browser cannot submit before configuring",
      empty.eval("llmRequire()") is False)
check("submitting without credentials opens the dialog but not the dock",
      empty.eval('$("llmmodal").hidden') is False
      and empty.eval('$("llmpanel").open') is False)
empty.eval("llmModalGo();")
check("the dialog's settings button opens the dock",
      empty.eval('$("llmpanel").open') is True
      and empty.eval('$("llmmodal").hidden') is True)
empty.eval("llmModalOpen(); llmModalClose();")
check("the dialog can be dismissed without opening the dock",
      empty.eval('$("llmmodal").hidden') is True
      and empty.eval('$("llmpanel").open') is True)

print()
print("FAILED: " + ", ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
