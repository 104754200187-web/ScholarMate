"""LLM provider selection and request shaping -- offline."""

import io
import os
import sys
import urllib.error

from _paths import ROOT  # noqa: F401  (puts the repository root on sys.path)
import llm


fails = []


def check(name, cond, extra=""):
    print(("  ok  " if cond else "  FAIL") + " " + name
          + ("" if cond else "  <- " + str(extra)))
    if not cond:
        fails.append(name)


KEYS = (
    "LLM_PROVIDER",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_STYLE",
    "OPENAI_MIN_OUTPUT_TOKENS",
    "OPENAI_REASONING_EFFORT",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "ZHIPU_API_KEY",
    "OPENROUTER_API_KEY",
    "COHERE_API_KEY",
    "BAIDU_API_KEY",
    "HUNYUAN_API_KEY",
)
saved_env = {key: os.environ.get(key) for key in KEYS}
real_load_dotenv = llm._load_dotenv
real_post_json = llm._post_json


def reset_env(**values):
    for key in KEYS:
        os.environ.pop(key, None)
    os.environ.update(values)


def capture_post(response):
    calls = []

    def fake_post(url, api_key, payload, label, headers=None):
        calls.append({"url": url, "api_key": api_key,
                      "payload": payload, "label": label,
                      "headers": headers})
        return response

    llm._post_json = fake_post
    return calls


llm._load_dotenv = lambda path=".env": None

try:
    # ---- 1. DeepSeek remains the default -----------------------------------
    reset_env(DEEPSEEK_API_KEY="ds-key")
    check("DeepSeek remains the default provider", llm.provider() == "deepseek")
    check("the UI label matches the provider", llm.provider_label() == "DeepSeek")

    calls = capture_post({"choices": [{"message": {"content": "deepseek reply"}}]})
    text = llm.chat([{"role": "user", "content": "hello"}],
                    temperature=0.1, max_tokens=768)
    call = calls[0]
    check("DeepSeek keeps its chat-completions endpoint",
          call["url"] == llm.DEEPSEEK_URL, call["url"])
    check("DeepSeek uses the configured key", call["api_key"] == "ds-key")
    check("DeepSeek defaults to deepseek-flash",
          call["payload"]["model"] == "deepseek-flash", call["payload"])
    check("DeepSeek keeps temperature and max_tokens",
          call["payload"]["temperature"] == 0.1
          and call["payload"]["max_tokens"] == 768, call["payload"])
    check("DeepSeek text is returned unchanged", text == "deepseek reply", text)

    calls = capture_post({"choices": [{"message": {"content": "ok"}}]})
    reset_env(DEEPSEEK_API_KEY="ds-key", DEEPSEEK_MODEL="deepseek-reasoner")
    llm.chat([{"role": "user", "content": "hello"}])
    check("DEEPSEEK_MODEL overrides the default model",
          calls[0]["payload"]["model"] == "deepseek-reasoner", calls[0]["payload"])

    # ---- 2. OpenAI Responses API ------------------------------------------
    reset_env(LLM_PROVIDER="openai", OPENAI_API_KEY="oa-key",
              OPENAI_MODEL="gpt-test", OPENAI_BASE_URL="https://gateway.test/v1/")
    check("openai selects the OpenAI label", llm.provider_label() == "OpenAI")

    calls = capture_post({"output_text": "responses reply"})
    messages = [
        {"role": "system", "content": "be precise"},
        {"role": "developer", "content": "return JSON"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    text = llm.chat(messages, temperature=0.9, max_tokens=512)
    call = calls[0]
    payload = call["payload"]
    check("Responses API is the OpenAI default",
          call["url"] == "https://gateway.test/v1/responses", call["url"])
    check("OpenAI uses the configured key", call["api_key"] == "oa-key")
    check("OpenAI model comes from OPENAI_MODEL",
          payload["model"] == "gpt-test", payload)
    check("system and developer messages become instructions",
          payload["instructions"] == "be precise\n\nreturn JSON", payload)
    check("only conversation messages become input",
          payload["input"] == [
              {"role": "user", "content": "hello"},
              {"role": "assistant", "content": "hi"},
          ], payload["input"])
    check("Responses API disables storage", payload["store"] is False, payload)
    check("Responses API does not send temperature",
          "temperature" not in payload, payload)
    check("small calls get the minimum output budget",
          payload["max_output_tokens"] == 2048, payload)
    check("the direct output_text field is returned",
          text == "responses reply", text)

    calls = capture_post({"output": [{
        "type": "message",
        "content": [{"type": "output_text", "text": "fallback text"}],
    }]})
    reset_env(LLM_PROVIDER="openai", OPENAI_API_KEY="oa-key")
    text = llm.chat([{"role": "user", "content": "hello"}])
    check("output[].content[].text is parsed when output_text is absent",
          text == "fallback text", text)

    calls = capture_post({"output_text": "ok"})
    reset_env(LLM_PROVIDER="openai", OPENAI_API_KEY="oa-key",
              OPENAI_MIN_OUTPUT_TOKENS="1024",
              OPENAI_REASONING_EFFORT="low")
    llm.chat([{"role": "user", "content": "hello"}], max_tokens=4096)
    payload = calls[0]["payload"]
    check("a larger max_tokens is not lowered by the floor",
          payload["max_output_tokens"] == 4096, payload)
    check("OPENAI_REASONING_EFFORT is forwarded",
          payload["reasoning"] == {"effort": "low"}, payload)

    calls = capture_post({"output_text": "ok"})
    reset_env(LLM_PROVIDER="gpt", OPENAI_API_KEY="oa-key", OPENAI_MODEL="env-model")
    llm.chat([{"role": "user", "content": "hello"}], model="explicit-model")
    check("gpt is accepted as an alias for openai",
          llm.provider() == "openai")
    check("an explicit model argument wins over OPENAI_MODEL",
          calls[0]["payload"]["model"] == "explicit-model", calls[0]["payload"])

    calls = capture_post({"output_text": "ok"})
    reset_env(LLM_PROVIDER="openai", OPENAI_API_KEY="oa-key",
              OPENAI_BASE_URL="https://gateway.test/v1")
    llm.chat([{"role": "user", "content": "hello"}])
    check("the default OpenAI model is gpt-6.1-sol",
          calls[0]["payload"]["model"] == "gpt-6.1-sol", calls[0]["payload"])

    # ---- 3. Chat Completions compatibility mode ---------------------------
    calls = capture_post({"choices": [{"message": {"content": "chat reply"}}]})
    reset_env(LLM_PROVIDER="openai", OPENAI_API_KEY="oa-key",
              OPENAI_BASE_URL="https://gateway.test/v1",
              OPENAI_API_STYLE="chat")
    text = llm.chat([{"role": "user", "content": "hello"}],
                    temperature=0.2, max_tokens=640)
    call = calls[0]
    payload = call["payload"]
    check("chat style calls /chat/completions",
          call["url"] == "https://gateway.test/v1/chat/completions", call["url"])
    check("chat style preserves messages and temperature",
          payload["messages"] == [{"role": "user", "content": "hello"}]
          and payload["temperature"] == 0.2, payload)
    check("chat style uses max_completion_tokens",
          payload["max_completion_tokens"] == 640
          and "max_tokens" not in payload, payload)
    check("chat style returns the assistant text", text == "chat reply", text)

    # ---- 4. A browser user's config wins over the server env --------------
    reset_env(DEEPSEEK_API_KEY="server-key")
    calls = capture_post({"choices": [{"message": {"content": "qwen reply"}}]})
    cfg = llm.normalize_config({
        "provider": "qwen", "api_key": "user-qwen-key", "model": "qwen-max",
    })
    with llm.use_config(cfg):
        check("the active config selects its provider", llm.provider() == "qwen")
        text = llm.chat([{"role": "user", "content": "hello"}],
                        temperature=0.4, max_tokens=700)
    call = calls[0]
    check("Qwen calls its OpenAI-compatible endpoint",
          call["url"] == llm.QWEN_BASE_URL + "/chat/completions", call["url"])
    check("the user's key is used, not the server key",
          call["api_key"] == "user-qwen-key", call["api_key"])
    check("the user's model is used", call["payload"]["model"] == "qwen-max")
    check("Qwen keeps max_tokens and temperature",
          call["payload"]["max_tokens"] == 700
          and call["payload"]["temperature"] == 0.4, call["payload"])
    check("Qwen returns the assistant text", text == "qwen reply", text)

    calls = capture_post({"choices": [{"message": {"content": "deepseek reply"}}]})
    cfg = llm.normalize_config({
        "provider": "deepseek", "api_key": "user-ds-key", "model": "deepseek-chat",
    })
    with llm.use_config(cfg):
        llm.chat([{"role": "user", "content": "hello"}], max_tokens=321)
    check("a user DeepSeek config uses its own key",
          calls[0]["api_key"] == "user-ds-key", calls[0]["api_key"])
    check("a user DeepSeek config keeps the fixed endpoint",
          calls[0]["url"] == llm.DEEPSEEK_URL, calls[0]["url"])

    calls = capture_post({"output_text": "user response"})
    cfg = llm.normalize_config({
        "provider": "openai", "api_key": "user-oa-key", "model": "gpt-user",
        "api_style": "responses", "min_output_tokens": 4096,
        "reasoning_effort": "medium",
    })
    with llm.use_config(cfg):
        llm.chat([{"role": "user", "content": "hello"}], max_tokens=64)
    payload = calls[0]["payload"]
    check("a user GPT config uses its own key",
          calls[0]["api_key"] == "user-oa-key", calls[0]["api_key"])
    check("a user GPT config keeps its model",
          payload["model"] == "gpt-user", payload)
    check("per-user output floor and reasoning effort are honoured",
          payload["max_output_tokens"] == 4096
          and payload["reasoning"] == {"effort": "medium"}, payload)
    check("the config context does not leak after the task",
          llm.current_config() is None)
    reset_env(DEEPSEEK_API_KEY="server-key")
    check("without a task config the server default is back",
          llm.provider() == "deepseek")

    check("Qwen defaults are filled in",
          llm.normalize_config({"provider": "qwen", "api_key": "k"})["model"]
          == "qwen-plus")

    # ---- 4b. Official providers and protocol adapters ---------------------
    defaults = {
        "deepseek": ("deepseek-flash", "chat", llm.DEEPSEEK_BASE_URL),
        "openai": ("gpt-6.1-sol", "responses", llm.DEFAULT_OPENAI_BASE_URL),
        "anthropic": ("claude-opus-5-5", "anthropic", llm.ANTHROPIC_BASE_URL),
        "gemini": ("gemini-3.8-flash", "gemini", llm.GEMINI_BASE_URL),
        "qwen": ("qwen-plus", "chat", llm.QWEN_BASE_URL),
        "xai": ("grok-4.7", "responses", llm.XAI_BASE_URL),
        "mistral": ("mistral-large-4", "chat", llm.MISTRAL_BASE_URL),
        "moonshot": ("kimi-k3", "chat", llm.MOONSHOT_BASE_URL),
        "zhipu": ("glm-5.3", "chat", llm.ZHIPU_BASE_URL),
        "doubao": ("doubao-seed-2-1-pro-260915", "chat", llm.DOUBAO_BASE_URL),
    }
    for name, (model, style, base) in defaults.items():
        cfg = llm.normalize_config({"provider": name, "api_key": "k"})
        check(f"{name} has the documented default model",
              cfg["model"] == model, cfg)
        check(f"{name} uses its fixed official endpoint and style",
              cfg["base_url"] == base and cfg["api_style"] == style, cfg)

    redirected = llm.normalize_config({
        "provider": "deepseek", "api_key": "k", "base_url": "https://evil.test/v1",
    })
    check("a browser cannot redirect a built-in provider endpoint",
          redirected["base_url"] == llm.DEEPSEEK_BASE_URL, redirected)

    calls = capture_post({"content": [
        {"type": "text", "text": "anthropic reply"},
    ]})
    cfg = llm.normalize_config({"provider": "anthropic", "api_key": "an-key"})
    with llm.use_config(cfg):
        text = llm.chat([
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "hello"},
        ], temperature=0.2, max_tokens=900)
    call = calls[0]
    check("Anthropic calls /v1/messages",
          call["url"] == llm.ANTHROPIC_BASE_URL + "/v1/messages", call["url"])
    check("Anthropic uses x-api-key rather than bearer auth",
          call["headers"] == {"x-api-key": "an-key",
                              "anthropic-version": llm.ANTHROPIC_VERSION},
          call["headers"])
    check("Anthropic puts system prompts at the top level",
          call["payload"]["system"] == "be precise"
          and call["payload"]["messages"] == [{"role": "user", "content": "hello"}],
          call["payload"])
    check("Anthropic response text blocks are joined",
          text == "anthropic reply", text)

    calls = capture_post({"steps": [{"type": "model_output", "content": [
        {"type": "text", "text": "gemini reply"},
    ]}]})
    cfg = llm.normalize_config({"provider": "gemini", "api_key": "gm-key"})
    with llm.use_config(cfg):
        text = llm.chat([
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ], temperature=0.9, max_tokens=777)
    call = calls[0]
    check("Gemini calls the Interactions API",
          call["url"] == llm.GEMINI_BASE_URL + "/interactions", call["url"])
    check("Gemini uses the x-goog-api-key header",
          call["headers"] == {"x-goog-api-key": "gm-key"}, call["headers"])
    check("Gemini passes model, system instruction and transcript",
          call["payload"]["model"] == "gemini-3.8-flash"
          and call["payload"]["system_instruction"] == "be precise"
          and "User: hello" in call["payload"]["input"]
          and "Assistant: hi" in call["payload"]["input"], call["payload"])
    check("Gemini output cap and no-storage flag are set",
          call["payload"]["generation_config"]["max_output_tokens"] == 777
          and call["payload"]["store"] is False, call["payload"])
    check("Gemini response steps are parsed", text == "gemini reply", text)

    calls = capture_post({"output_text": "grok reply"})
    cfg = llm.normalize_config({"provider": "xai", "api_key": "x-key"})
    with llm.use_config(cfg):
        text = llm.chat([{"role": "user", "content": "hello"}])
    check("xAI uses the official Responses endpoint",
          calls[0]["url"] == llm.XAI_BASE_URL + "/responses", calls[0]["url"])
    check("xAI defaults to grok-4.7",
          calls[0]["payload"]["model"] == "grok-4.7", calls[0]["payload"])
    check("xAI returns response text", text == "grok reply", text)

    for name, model, endpoint in (
        ("mistral", "mistral-large-4", llm.MISTRAL_BASE_URL + "/chat/completions"),
        ("moonshot", "kimi-k3", llm.MOONSHOT_BASE_URL + "/chat/completions"),
        ("zhipu", "glm-5.3", llm.ZHIPU_BASE_URL + "/chat/completions"),
        ("doubao", "doubao-seed-2-1-pro-260915",
         llm.DOUBAO_BASE_URL + "/chat/completions"),
        ("baidu", "ERNIE-5.1", llm.BAIDU_BASE_URL + "/chat/completions"),
        ("hunyuan", "hunyuan-turbos-latest",
         llm.HUNYUAN_BASE_URL + "/chat/completions"),
        ("minimax", "MiniMax-M3.1-Flash",
         llm.MINIMAX_BASE_URL + "/chat/completions"),
        ("groq", "openai/gpt-oss-120b", llm.GROQ_BASE_URL + "/chat/completions"),
        ("openrouter", "~openai/gpt-sol-latest",
         llm.OPENROUTER_BASE_URL + "/chat/completions"),
        ("cohere", "command-a-plus-05-2026",
         llm.COHERE_BASE_URL + "/chat/completions"),
        ("siliconflow", "deepseek-ai/DeepSeek-V4-Flash",
         llm.SILICONFLOW_BASE_URL + "/chat/completions"),
    ):
        calls = capture_post({"choices": [{"message": {"content": name + " reply"}}]})
        cfg = llm.normalize_config({"provider": name, "api_key": name + "-key"})
        with llm.use_config(cfg):
            text = llm.chat([{"role": "user", "content": "hello"}], max_tokens=600)
        check(f"{name} uses its official OpenAI-compatible endpoint",
              calls[0]["url"] == endpoint, calls[0]["url"])
        check(f"{name} uses its documented model and bearer auth",
              calls[0]["payload"]["model"] == model
              and calls[0]["headers"] is None, calls[0])
        check(f"{name} returns text", text == name + " reply", text)
    for bad in (
        "http://gateway.example/v1",
        "https://localhost/v1",
        "https://127.0.0.1/v1",
        "https://[::1]/v1",
        "https://10.0.0.1/v1",
        "https://169.254.169.254/v1",
        "https://metadata.google.internal/v1",
        "https://example.com:8443/v1",
        "https://user:pass@example.com/v1",
    ):
        try:
            llm.normalize_config({"provider": "custom", "api_key": "k",
                                  "model": "m", "base_url": bad})
            check("unsafe base URL rejected: " + bad, False, "accepted")
        except ValueError:
            check("unsafe base URL rejected: " + bad, True)
    try:
        llm.normalize_config({"provider": "custom", "api_key": "k", "model": "m"})
        check("custom service requires a base URL", False, "accepted")
    except ValueError:
        check("custom service requires a base URL", True)
    try:
        llm.normalize_config({"provider": "qwen"})
        check("a user config without a key is rejected", False, "accepted")
    except ValueError:
        check("a user config without a key is rejected", True)

    # ---- 4b. Server-owned free-tier access --------------------------------
    reset_env()
    check("no free-tier keys means no free options", llm.free_options() == [])

    reset_env(GEMINI_API_KEY="gm-free-key", GROQ_API_KEY="")
    options = llm.free_options()
    check("only providers with a configured key are offered",
          [opt["provider"] for opt in options] == ["gemini"], options)
    check("free options name the fixed model and note",
          options[0]["model"] == "gemini-3.8-flash"
          and "免费层" in options[0]["note"], options)
    check("the free options payload never contains the server key",
          all("key" not in opt and "api_key" not in opt for opt in options), options)

    cfg = llm.free_config_for("gemini", "gemini-3.8-flash")
    check("free_config_for builds a server-owned config",
          cfg["provider"] == "gemini" and cfg["free_access"] is True
          and cfg["api_key"] == "gm-free-key", cfg)
    check("free configs are visibly marked in the run label",
          llm.config_label(cfg).startswith("免费体验 · "), llm.config_label(cfg))
    check("free configs keep the official endpoint",
          cfg["base_url"] == llm.GEMINI_BASE_URL and cfg["api_style"] == "gemini", cfg)

    calls = capture_post({"steps": [{"type": "model_output", "content": [
        {"type": "text", "text": "free reply"},
    ]}]})
    with llm.use_config(cfg):
        text = llm.chat([{"role": "user", "content": "hello"}])
    check("a free config calls the provider with the server key",
          calls[0]["api_key"] == "gm-free-key"
          and calls[0]["headers"] == {"x-goog-api-key": "gm-free-key"}, calls[0])
    check("a free config returns the model text", text == "free reply", text)

    reset_env(GEMINI_API_KEY="gm-free-key", GROQ_API_KEY="groq-free-key",
              ZHIPU_API_KEY="zhipu-free-key",
              OPENROUTER_API_KEY="or-free-key", COHERE_API_KEY="co-free-key",
              BAIDU_API_KEY="baidu-free-key", HUNYUAN_API_KEY="hunyuan-free-key")
    for name, model, endpoint, env_name in (
        ("groq", "openai/gpt-oss-120b",
         llm.GROQ_BASE_URL + "/chat/completions", "GROQ_API_KEY"),
        ("zhipu", "GLM-4.7-Flash",
         llm.ZHIPU_BASE_URL + "/chat/completions", "ZHIPU_API_KEY"),
        ("openrouter", "minimax/minimax-m3.1-flash:free",
         llm.OPENROUTER_BASE_URL + "/chat/completions", "OPENROUTER_API_KEY"),
        ("cohere", "command-a-plus-05-2026",
         llm.COHERE_BASE_URL + "/chat/completions", "COHERE_API_KEY"),
        ("baidu", "ERNIE-4.5-Turbo-128K",
         llm.BAIDU_BASE_URL + "/chat/completions", "BAIDU_API_KEY"),
        ("hunyuan", "hunyuan-a13b",
         llm.HUNYUAN_BASE_URL + "/chat/completions", "HUNYUAN_API_KEY"),
    ):
        calls = capture_post({"choices": [{"message": {"content": name + " free"}}]})
        free_cfg = llm.free_config_for(name, model)
        with llm.use_config(free_cfg):
            text = llm.chat([{"role": "user", "content": "hello"}])
        check(f"free access uses {name}'s official endpoint",
              calls[0]["url"] == endpoint, calls[0]["url"])
        check(f"free access pins {name}'s allow-listed model",
              calls[0]["payload"]["model"] == model
              and calls[0]["api_key"] == os.environ[env_name], calls[0])
        check(f"free access returns {name}'s text", text == name + " free", text)

    for name, model in (("gemini", "gemini-4-argon"),
                        ("openai", "gpt-6.1-sol"),
                        ("deepseek", "deepseek-flash")):
        try:
            llm.free_config_for(name, model)
            check("free access rejects a non-allow-listed model: " + name,
                  False, "accepted")
        except ValueError:
            check("free access rejects a non-allow-listed model: " + name, True)

    reset_env()
    try:
        llm.free_config_for("gemini")
        missing_free_key = None
    except RuntimeError as exc:
        missing_free_key = str(exc)
    check("an unconfigured free provider fails loudly",
          missing_free_key and "管理员尚未配置" in missing_free_key,
          missing_free_key)

    # ---- 5. Configuration errors are explicit, never silent ---------------
    reset_env(LLM_PROVIDER="openai")
    try:
        llm.chat([{"role": "user", "content": "hello"}])
        missing_key = None
    except RuntimeError as exc:
        missing_key = str(exc)
    check("a missing OPENAI_API_KEY is reported",
          missing_key and "OPENAI_API_KEY not set" in missing_key, missing_key)

    reset_env(LLM_PROVIDER="openai", OPENAI_API_KEY="oa-key",
              OPENAI_API_STYLE="unknown")
    try:
        llm.chat([{"role": "user", "content": "hello"}])
        bad_style = None
    except RuntimeError as exc:
        bad_style = str(exc)
    check("an unknown OPENAI_API_STYLE is rejected",
          bad_style and "OPENAI_API_STYLE" in bad_style, bad_style)

    reset_env(LLM_PROVIDER="anthropic")
    try:
        llm.provider()
        bad_provider = None
    except RuntimeError as exc:
        bad_provider = str(exc)
    check("an unknown LLM_PROVIDER is rejected",
          bad_provider and "LLM_PROVIDER" in bad_provider, bad_provider)

    llm._post_json = real_post_json
    attempts = []
    real_open = llm._open
    real_sleep = llm.time.sleep

    def quota_open(req):
        attempts.append(req.full_url)
        body = (b'{"error":{"type":"insufficient_quota",'
                b'"code":"credit_balance_exhausted"}}')
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests",
                                     {}, io.BytesIO(body))

    llm._open = quota_open
    llm.time.sleep = lambda seconds: None
    try:
        llm._post_json("https://api.openai.com/v1/responses", "oa-key", {}, "OpenAI")
        quota_error = None
    except RuntimeError as exc:
        quota_error = str(exc)
    finally:
        llm._open = real_open
        llm.time.sleep = real_sleep
    check("an exhausted balance is not retried",
          len(attempts) == 1, attempts)
    check("the quota error names the real cause",
          quota_error and "insufficient_quota" in quota_error, quota_error)

    check("parse_json still tolerates prose around JSON",
          llm.parse_json('result: {"ok": true} trailing') == {"ok": True})
finally:
    llm._load_dotenv = real_load_dotenv
    llm._post_json = real_post_json
    for key, value in saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


print()
print("FAILED: " + ", ".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
