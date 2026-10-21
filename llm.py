"""DeepSeek LLM wrapper (OpenAI-compatible API, pure stdlib).

Set DEEPSEEK_API_KEY in the environment or in a .env file.
"""

import json
import os
import re
import time
import urllib.request
import urllib.error

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def chat(messages, model=DEFAULT_MODEL, temperature=0.3, max_tokens=2048):
    _load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Put it in the environment or a .env file."
        )
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        DEEPSEEK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            last_err = e
            if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504):
                raise
            # say so: a retry storm is invisible otherwise, and it is what a
            # slow stage under a flaky network actually is (a four-call probe
            # of this endpoint failed once with an SSL EOF while the
            # per-sentence stage was being timed, and each retry costs 2-6s)
            print(f"      [warn] DeepSeek call failed ({type(e).__name__}: {e}); "
                  f"retry {attempt + 2}/3")
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"DeepSeek request failed after 3 attempts: {last_err}")
    return data["choices"][0]["message"]["content"]


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
