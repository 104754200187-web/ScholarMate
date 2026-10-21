"""LLM wrapper for major hosted LLMs and OpenAI-compatible services (stdlib).

The Web UI supplies one validated config per browser/task through
``use_config()``; the key never enters the job store. CLI/server calls fall
back to the environment:

    LLM_PROVIDER=deepseek                 # default; DEEPSEEK_API_KEY
    LLM_PROVIDER=openai                   # OPENAI_API_KEY

Browser configs cover DeepSeek, OpenAI, Anthropic, Gemini, Qwen, xAI, Mistral,
Cohere, Groq, OpenRouter, Baidu ERNIE, Tencent Hunyuan, MiniMax, Moonshot/Kimi,
Zhipu/GLM, Doubao and SiliconFlow. Provider endpoints and default model IDs
follow the official API documentation as of 2026-10-21; the model field remains
editable because vendors rename and retire models.
"""

import json
import os
import re
import time
import contextlib
import contextvars
import ipaddress
import urllib.error
import urllib.parse
import urllib.request

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_URL = DEEPSEEK_BASE_URL + "/chat/completions"
DEFAULT_MODEL = "deepseek-flash"             # backwards-compatible name
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_OPENAI_MODEL = "gpt-6.1-sol"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_QWEN_MODEL = "qwen-plus"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_XAI_MODEL = "grok-4.7"
XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MISTRAL_MODEL = "mistral-large-4"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MOONSHOT_MODEL = "kimi-k3"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_ZHIPU_MODEL = "glm-5.3"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_DOUBAO_MODEL = "doubao-seed-2-1-pro-260915"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_BAIDU_MODEL = "ERNIE-5.1"
BAIDU_BASE_URL = "https://qianfan.baidubce.com/v2"
DEFAULT_HUNYUAN_MODEL = "hunyuan-turbos-latest"
HUNYUAN_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1"
DEFAULT_MINIMAX_MODEL = "MiniMax-M3.1-Flash"
MINIMAX_BASE_URL = "https://api.minimax.cn/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_OPENROUTER_MODEL = "~openai/gpt-sol-latest"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_COHERE_MODEL = "command-a-plus-05-2026"
COHERE_BASE_URL = "https://api.cohere.ai/compatibility/v1"
DEFAULT_SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

PROVIDER_LABELS = {
    "deepseek": "DeepSeek",
    "openai": "OpenAI GPT",
    "anthropic": "Anthropic Claude",
    "gemini": "Google Gemini",
    "qwen": "通义千问",
    "xai": "xAI Grok",
    "mistral": "Mistral AI",
    "moonshot": "Moonshot Kimi",
    "zhipu": "智谱 GLM",
    "doubao": "字节豆包 Doubao",
    "baidu": "百度文心 ERNIE",
    "hunyuan": "腾讯混元 Hunyuan",
    "minimax": "MiniMax",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "cohere": "Cohere",
    "siliconflow": "硅基流动 SiliconFlow",
    "custom": "自定义 OpenAI 兼容服务",
}

# Official defaults verified against each vendor's API documentation. Browser
# requests may override only the model, never the provider endpoint.
PROVIDER_DEFAULTS = {
    "deepseek": (DEFAULT_DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, "chat"),
    "openai": (DEFAULT_OPENAI_MODEL, DEFAULT_OPENAI_BASE_URL, "responses"),
    "anthropic": (DEFAULT_ANTHROPIC_MODEL, ANTHROPIC_BASE_URL, "anthropic"),
    "gemini": (DEFAULT_GEMINI_MODEL, GEMINI_BASE_URL, "gemini"),
    "qwen": (DEFAULT_QWEN_MODEL, QWEN_BASE_URL, "chat"),
    "xai": (DEFAULT_XAI_MODEL, XAI_BASE_URL, "responses"),
    "mistral": (DEFAULT_MISTRAL_MODEL, MISTRAL_BASE_URL, "chat"),
    "moonshot": (DEFAULT_MOONSHOT_MODEL, MOONSHOT_BASE_URL, "chat"),
    "zhipu": (DEFAULT_ZHIPU_MODEL, ZHIPU_BASE_URL, "chat"),
    "doubao": (DEFAULT_DOUBAO_MODEL, DOUBAO_BASE_URL, "chat"),
    "baidu": (DEFAULT_BAIDU_MODEL, BAIDU_BASE_URL, "chat"),
    "hunyuan": (DEFAULT_HUNYUAN_MODEL, HUNYUAN_BASE_URL, "chat"),
    "minimax": (DEFAULT_MINIMAX_MODEL, MINIMAX_BASE_URL, "chat"),
    "groq": (DEFAULT_GROQ_MODEL, GROQ_BASE_URL, "chat"),
    "openrouter": (DEFAULT_OPENROUTER_MODEL, OPENROUTER_BASE_URL, "chat"),
    "cohere": (DEFAULT_COHERE_MODEL, COHERE_BASE_URL, "chat"),
    "siliconflow": (DEFAULT_SILICONFLOW_MODEL, SILICONFLOW_BASE_URL, "chat"),
}

# Server-owned "free experience" models. These are the only models a browser
# may run without supplying its own key: each one is either on a vendor free
# tier or a trial key with a documented monthly cap. The server key never
# leaves the process, and browsers cannot redirect these requests.
FREE_ACCESS = {
    "gemini": {
        "env": "GEMINI_API_KEY", "model": DEFAULT_GEMINI_MODEL,
        "note": "Google 免费层",
    },
    "groq": {
        "env": "GROQ_API_KEY", "model": DEFAULT_GROQ_MODEL,
        "note": "Groq Free Plan",
    },
    "zhipu": {
        "env": "ZHIPU_API_KEY", "model": "GLM-4.7-Flash",
        "note": "智谱免费文本模型",
    },
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "model": "minimax/minimax-m3.1-flash:free",
        "note": "OpenRouter 免费模型",
    },
    "cohere": {
        "env": "COHERE_API_KEY", "model": DEFAULT_COHERE_MODEL,
        "note": "Cohere Trial Key（约 1000 次/月）",
    },
    "baidu": {
        "env": "BAIDU_API_KEY", "model": "ERNIE-4.5-Turbo-128K",
        "note": "百度新用户免费额度（100 万 tokens / 3 个月）",
    },
    "hunyuan": {
        "env": "HUNYUAN_API_KEY", "model": "hunyuan-a13b",
        "note": "腾讯混元新用户免费额度（100 万 tokens / 1 年）",
    },
}

# The web layer sets this for one job. It has to be a ContextVar: two workers
# may call the LLM at the same time with different users' keys.
_active_config = contextvars.ContextVar("llm_active_config", default=None)

_LOCAL_HOST_SUFFIXES = (
    ".localhost", ".local", ".internal", ".home.arpa",
    ".nip.io", ".sslip.io", ".xip.io", ".localtest.me", ".lvh.me", ".vcap.me",
)


def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


@contextlib.contextmanager
def use_config(config):
    """Use one browser user's LLM config for this thread's task."""
    token = _active_config.set(config)
    try:
        yield
    finally:
        _active_config.reset(token)


def current_config():
    return _active_config.get()


def server_default_allowed():
    """Whether a task may fall back to the keys in the server's ``.env``."""
    _load_dotenv()
    return (os.environ.get("LLM_ALLOW_SERVER_DEFAULT") or "0").strip() == "1"


def _env_int(name, default):
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _validate_public_https(url, base_only=False):
    """Reject schemes, credentials and hosts that could reach the local network.

    A user-supplied base URL turns the server into their HTTP client, so this is
    an SSRF boundary, not cosmetic validation. Redirects are checked again by
    ``_SafeApiRedirectHandler`` below.
    """
    if not isinstance(url, str):
        raise ValueError("Base URL 必须是字符串")
    url = url.strip()
    if not url or len(url) > 1000:
        raise ValueError("Base URL 为空或过长")
    if any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise ValueError("Base URL 含有控制字符")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() != "https":
        raise ValueError("Base URL 必须使用 https://")
    if parts.username or parts.password:
        raise ValueError("Base URL 不能包含用户名或密码")
    host = (parts.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError("Base URL 没有主机名")
    if host == "localhost" or host.endswith(_LOCAL_HOST_SUFFIXES):
        raise ValueError(f"不允许访问本地主机：{host}")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"Base URL 端口无效：{exc}") from None
    if port not in (None, 443):
        raise ValueError(f"Base URL 只允许 443 端口：{port}")
    numeric = re.fullmatch(r"[0-9.]+", host) or host.startswith("0x")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if numeric or ":" in host or "%" in host or re.search(r"\s", host):
            raise ValueError(f"Base URL 主机名不明确：{host}") from None
    else:
        if not address.is_global:
            raise ValueError(f"Base URL 不允许内网地址：{host}")
    if base_only and (parts.query or parts.fragment):
        raise ValueError("Base URL 不能带查询串或片段")
    return url.rstrip("/") if base_only else url


def normalize_config(raw, require_key=True):
    """Validate one browser-supplied LLM config and fill provider defaults."""
    if not isinstance(raw, dict):
        raise ValueError("LLM 配置必须是 JSON 对象")
    _load_dotenv()
    name = str(raw.get("provider") or "deepseek").strip().lower()
    if name == "gpt":
        name = "openai"
    if name not in PROVIDER_LABELS:
        raise ValueError(f"未知的模型服务：{name}")

    api_key = str(raw.get("api_key") or "").strip()
    if len(api_key) > 8192:
        raise ValueError("API Key 过长")
    if require_key and not api_key:
        raise ValueError("API Key 不能为空")
    model = str(raw.get("model") or "").strip()[:200]
    base = str(raw.get("base_url") or "").strip()
    style = str(raw.get("api_style") or "").strip().lower()

    if name == "custom":
        if not model:
            raise ValueError("自定义模型服务需要填写模型名")
        if not base:
            raise ValueError("自定义模型服务需要填写 Base URL")
        style = style or "chat"
    else:
        default_model, official_base, official_style = PROVIDER_DEFAULTS[name]
        model = model or default_model
        # Built-in providers cannot be redirected by a browser. A custom
        # gateway belongs to the explicit "custom" provider.
        base = official_base
        style = (style or official_style) if name == "openai" else official_style

    base = _validate_public_https(base, base_only=True)
    if style not in ("responses", "chat", "anthropic", "gemini"):
        raise ValueError(f"未知的 API 风格：{style}")
    if name == "openai" and style not in ("responses", "chat"):
        raise ValueError("OpenAI 只支持 Responses API 或 Chat Completions")
    if name in ("anthropic", "gemini") and style != PROVIDER_DEFAULTS[name][2]:
        raise ValueError(f"{PROVIDER_LABELS[name]} 不支持该 API 风格")
    if name == "custom" and style not in ("chat", "responses"):
        raise ValueError("自定义服务只支持 Chat Completions 或 Responses API")
    if name not in ("openai", "custom", "xai") and style == "responses":
        raise ValueError("该模型服务不支持 Responses API")

    floor = raw.get("min_output_tokens")
    if floor is None:
        floor = _env_int("OPENAI_MIN_OUTPUT_TOKENS", 2048)
    else:
        try:
            floor = max(0, int(floor))
        except (TypeError, ValueError):
            raise ValueError("min_output_tokens 必须是整数") from None
    effort = str(raw.get("reasoning_effort") or "").strip().lower()

    return {
        "provider": name,
        "api_key": api_key,
        "model": model,
        "base_url": base,
        "api_style": style,
        "min_output_tokens": floor,
        "reasoning_effort": effort,
    }


def config_label(config):
    if not config:
        return "服务器默认"
    provider = config.get("provider") or "deepseek"
    label = PROVIDER_LABELS.get(provider, provider)
    model = config.get("model") or ""
    text = f"{label} / {model}" if model else label
    return f"免费体验 · {text}" if config.get("free_access") else text


def free_options():
    """Free-tier models this server can actually serve (keys are not exposed)."""
    _load_dotenv()
    options = []
    for name, spec in FREE_ACCESS.items():
        if (os.environ.get(spec["env"]) or "").strip():
            options.append({
                "provider": name,
                "model": spec["model"],
                "label": PROVIDER_LABELS.get(name, name),
                "note": spec["note"],
            })
    return options


def free_config_for(name, model=None):
    """Build a server-owned config for one allow-listed free model.

    ``model`` is accepted only to reject tampering; it can never select an
    arbitrary paid model on the server's key.
    """
    _load_dotenv()
    name = str(name or "").strip().lower()
    spec = FREE_ACCESS.get(name)
    if spec is None:
        raise ValueError("该服务不在免费体验范围内")
    requested = str(model or "").strip()
    if requested and requested != spec["model"]:
        raise ValueError("免费体验只允许使用官方指定的免费模型")
    api_key = (os.environ.get(spec["env"]) or "").strip()
    if not api_key:
        raise RuntimeError("管理员尚未配置该免费体验服务的 API Key")
    config = normalize_config(
        {"provider": name, "api_key": api_key, "model": spec["model"]},
        require_key=True,
    )
    config["free_access"] = True
    config["free_note"] = spec["note"]
    return config


def provider(config=None):
    """Configured provider for a browser task, or the CLI/server default."""
    config = config if config is not None else _active_config.get()
    if config:
        return config.get("provider") or "deepseek"
    _load_dotenv()
    name = (os.environ.get("LLM_PROVIDER") or "deepseek").strip().lower()
    if name == "gpt":
        return "openai"
    if name not in ("deepseek", "openai"):
        raise RuntimeError(
            f"unknown LLM_PROVIDER={name!r}; use deepseek or openai")
    return name


def provider_label(config=None):
    """Short label for user-facing UI copy."""
    config = config if config is not None else _active_config.get()
    if config:
        return PROVIDER_LABELS.get(config.get("provider"), config.get("provider"))
    return "OpenAI" if provider() == "openai" else "DeepSeek"


class _SafeApiRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate every redirect hop before the server follows it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(req):
    opener = urllib.request.build_opener(_SafeApiRedirectHandler())
    return opener.open(req, timeout=120)


def _post_json(url, api_key, payload, label, headers=None):
    """POST JSON with the retry policy shared by all providers.

    ``headers`` replaces the default bearer auth for APIs such as Anthropic
    (``x-api-key``) and Gemini (``x-goog-api-key``).
    """
    request_headers = {"Content-Type": "application/json"}
    if headers is None:
        request_headers["Authorization"] = f"Bearer {api_key}"
    else:
        request_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
    )
    last_err = None
    for attempt in range(3):
        try:
            with _open(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")
            except Exception:
                detail = ""
            last_err = f"HTTP {e.code}" + (f": {detail[:400]}" if detail else "")
            retryable = e.code in (429, 500, 502, 503, 504)
            if e.code == 429 and detail:
                # OpenAI reports an exhausted balance as HTTP 429. Retrying it
                # only adds delay and can surface a misleading later error.
                try:
                    error = (json.loads(detail).get("error") or {})
                except (ValueError, AttributeError):
                    error = {}
                retryable = not (
                    error.get("type") == "insufficient_quota"
                    or error.get("code") == "credit_balance_exhausted"
                )
            if not retryable:
                raise RuntimeError(f"{label} request failed: {last_err}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < 2:
            # A retry storm is invisible otherwise. Keep the same 3-attempt
            # policy as the original DeepSeek wrapper.
            print(f"      [warn] {label} call failed ({last_err}); "
                  f"retry {attempt + 2}/3")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{label} request failed after 3 attempts: {last_err}")


def _chat_deepseek(messages, model, temperature, max_tokens):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Put it in the environment or a .env file.")
    payload = {
        "model": model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = _post_json(DEEPSEEK_URL, api_key, payload, "DeepSeek")
    return data["choices"][0]["message"]["content"]


def _openai_settings(model):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Put it in the environment or a .env file.")
    base = (os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    style = (os.environ.get("OPENAI_API_STYLE") or "responses").strip().lower()
    if style not in ("responses", "chat"):
        raise RuntimeError(
            f"unknown OPENAI_API_STYLE={style!r}; use responses or chat")
    model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL
    return api_key, base, style, model


def _responses_payload(messages, model, max_tokens, floor=None, effort=None):
    instructions = []
    inputs = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if role in ("system", "developer"):
            instructions.append(content)
        else:
            inputs.append({"role": role, "content": content})
    # max_output_tokens includes reasoning tokens on reasoning models. Small
    # verification calls (512/1024) would otherwise risk producing no text.
    floor = (_env_int("OPENAI_MIN_OUTPUT_TOKENS", 2048)
             if floor is None else max(0, int(floor)))
    payload = {
        "model": model,
        "input": inputs,
        "max_output_tokens": max(int(max_tokens), floor),
        "store": False,
    }
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    if effort is None:
        effort = (os.environ.get("OPENAI_REASONING_EFFORT") or "").strip().lower()
    else:
        effort = str(effort or "").strip().lower()
    if effort:
        payload["reasoning"] = {"effort": effort}
    return payload


def _responses_text(data):
    direct = data.get("output_text") if isinstance(data, dict) else None
    if isinstance(direct, str) and direct.strip():
        return direct
    parts = []
    for item in (data.get("output") or []) if isinstance(data, dict) else []:
        if item.get("type") not in (None, "message"):
            continue
        for content in item.get("content") or []:
            if content.get("type") in ("output_text", "text"):
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError("OpenAI response contained no output text")
    return text


def _chat_openai(messages, model, temperature, max_tokens):
    api_key, base, style, model = _openai_settings(model)
    if style == "responses":
        payload = _responses_payload(messages, model, max_tokens)
        data = _post_json(base + "/responses", api_key, payload, "OpenAI")
        return _responses_text(data)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    data = _post_json(base + "/chat/completions", api_key, payload, "OpenAI")
    return data["choices"][0]["message"]["content"]


def _anthropic_payload(messages, model, temperature, max_tokens):
    system = []
    turns = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if role in ("system", "developer"):
            system.append(content)
        else:
            turns.append({"role": role if role == "assistant" else "user",
                          "content": content})
    payload = {
        "model": model,
        "max_tokens": max(1, int(max_tokens)),
        "messages": turns,
    }
    if system:
        payload["system"] = "\n\n".join(system)
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def _anthropic_text(data, label):
    parts = []
    for block in (data.get("content") or []) if isinstance(data, dict) else []:
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError(f"{label} response contained no output text")
    return text


def _gemini_payload(messages, model, max_tokens):
    system = []
    turns = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if role in ("system", "developer"):
            system.append(content)
        else:
            turns.append(("Assistant" if role == "assistant" else "User", content))
    payload = {
        "model": model,
        "input": "\n\n".join(f"{role}: {content}" for role, content in turns),
        "generation_config": {"max_output_tokens": max(1, int(max_tokens))},
        "store": False,
    }
    if system:
        payload["system_instruction"] = "\n\n".join(system)
    return payload


def _gemini_text(data, label):
    direct = data.get("output_text") if isinstance(data, dict) else None
    if isinstance(direct, str) and direct.strip():
        return direct
    parts = []
    for step in (data.get("steps") or []) if isinstance(data, dict) else []:
        if step.get("type") != "model_output":
            continue
        for content in step.get("content") or []:
            if content.get("type") == "text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError(f"{label} response contained no output text")
    return text


def _chat_anthropic(messages, config, model, temperature, max_tokens):
    label = PROVIDER_LABELS[config["provider"]]
    payload = _anthropic_payload(messages, model, temperature, max_tokens)
    data = _post_json(
        config["base_url"] + "/v1/messages", config["api_key"], payload, label,
        headers={"x-api-key": config["api_key"],
                 "anthropic-version": ANTHROPIC_VERSION},
    )
    return _anthropic_text(data, label)


def _chat_gemini(messages, config, model, max_tokens):
    label = PROVIDER_LABELS[config["provider"]]
    payload = _gemini_payload(messages, model, max_tokens)
    data = _post_json(
        config["base_url"] + "/interactions", config["api_key"], payload, label,
        headers={"x-goog-api-key": config["api_key"]},
    )
    return _gemini_text(data, label)


def _chat_configured(messages, config, model, temperature, max_tokens):
    """Call using one browser user's validated config and key."""
    name = config["provider"]
    label = PROVIDER_LABELS.get(name, name)
    model = model or config.get("model")
    api_key = config["api_key"]
    base = config["base_url"]
    style = config.get("api_style") or "chat"
    if style == "anthropic":
        return _chat_anthropic(messages, config, model, temperature, max_tokens)
    if style == "gemini":
        return _chat_gemini(messages, config, model, max_tokens)
    if style == "responses":
        payload = _responses_payload(
            messages, model, max_tokens,
            floor=config.get("min_output_tokens"),
            effort=config.get("reasoning_effort"),
        )
        data = _post_json(base + "/responses", api_key, payload, label)
        return _responses_text(data)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    # OpenAI reasoning models use max_completion_tokens; other OpenAI-
    # compatible gateways still accept max_tokens.
    payload["max_completion_tokens" if name == "openai" else "max_tokens"] = max_tokens
    data = _post_json(base + "/chat/completions", api_key, payload, label)
    return data["choices"][0]["message"]["content"]


def chat(messages, model=None, temperature=0.3, max_tokens=2048):
    """Call the configured provider and return the assistant text."""
    config = _active_config.get()
    if config:
        return _chat_configured(messages, config, model, temperature, max_tokens)
    _load_dotenv()
    name = provider()
    if name == "openai":
        return _chat_openai(messages, model, temperature, max_tokens)
    return _chat_deepseek(messages, model, temperature, max_tokens)


def chat_json(messages, **kwargs):
    """Call chat() and parse the reply as JSON, tolerating surrounding text."""
    return parse_json(chat(messages, **kwargs))


def _find_balanced(text, open_ch, close_ch):
    """Return the first balanced open..close span in text (JSON-aware)."""
    start = text.find(open_ch)
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find(open_ch, start + 1)
    return None


def parse_json(content):
    """Parse a model reply as JSON, tolerating markdown fences, prose around
    the JSON, and trailing text (balanced extraction instead of greedy
    {.*})."""
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    for candidate in (_find_balanced(content, "{", "}"),
                      _find_balanced(content, "[", "]")):
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"model did not return JSON: {content[:300]}")


def extract_verdicts(text, id_field="n"):
    """Tolerantly parse a list of {id_field, supported, reason} verdicts from a
    model reply, even when the JSON is malformed (e.g. quotes inside reasons)."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("verdicts")
        if isinstance(data, list):
            return data
    except Exception:
        pass
    out = []
    for m in re.finditer(r'"%s"\s*:\s*(\d+)' % re.escape(id_field), text):
        seg = text[m.start():m.start() + 400]
        sup = re.search(r'"supported"\s*:\s*(true|false)', seg)
        reason = re.search(r'"reason"\s*:\s*"([^"]*)"', seg)
        item = {
            id_field: int(m.group(1)),
            "reason": reason.group(1) if reason else "",
        }
        # A missing supported field is not a negative verdict. Leaving it out
        # lets judge_claims() mark the claim as unjudged instead of unsupported.
        if sup:
            item["supported"] = sup.group(1) == "true"
        out.append(item)
    return out
