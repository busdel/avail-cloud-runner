# -*- coding: utf-8 -*-
"""Minimal provider-agnostic LLM client (OpenAI-compatible chat completions).

Works with DeepSeek, OpenAI, Groq, and any endpoint that implements the
``POST /chat/completions`` OpenAI schema. No third-party SDK required.

Configuration (in priority order):
  1. explicit constructor args
  2. environment variables AVAIL_LLM_API_KEY / AVAIL_LLM_BASE_URL / AVAIL_LLM_MODEL
"""
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

PROVIDERS = {
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "qwen/qwen3.8-27b"},
    # Google Gemini via its OpenAI-compatible endpoint (no SDK needed).
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "model": "gemini-3.6-flash"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "model": "deepseek/deepseek-chat-v3:free"},
}


def load_api_key():
    """Resolve the API key: env var first, then ~/.config/avail/llm_key."""
    key = os.environ.get("AVAIL_LLM_API_KEY")
    if key:
        return key.strip()
    kf = Path.home() / '.config' / 'avail' / 'llm_key'
    if kf.exists():
        return kf.read_text().strip()
    return None


def load_api_key_for(provider):
    """Key for a named provider: env AVAIL_LLM_API_KEY_<PROVIDER> or key file."""
    key = os.environ.get(f"AVAIL_LLM_API_KEY_{provider.upper()}")
    if key:
        return key.strip()
    kf = Path.home() / '.config' / 'avail' / f'llm_key_{provider}'
    if kf.exists():
        return kf.read_text().strip()
    return None


class QuotaError(RuntimeError):
    """Raised when the provider rejects the request due to quota/rate limits."""


def build_clients(fallback=None):
    """Primary client + optional fallback providers (tried on quota errors).

    Fallback spec: comma-separated 'provider' names, e.g. 'groq,openrouter'.
    Keys are read from AVAIL_LLM_API_KEY_<PROVIDER> or key files.
    If no explicit spec is given, any ~/.config/avail/llm_key_<provider>
    files are auto-detected as fallbacks.
    """
    clients = []
    primary = LLMClient()
    clients.append(primary)
    spec = fallback or os.environ.get("AVAIL_LLM_FALLBACK", "")
    if not spec:
        cfg = Path.home() / '.config' / 'avail'
        if cfg.exists():
            spec = ",".join(sorted(p.stem.replace("llm_key_", "")
                                   for p in cfg.glob("llm_key_*")))
    for prov in [p.strip().lower() for p in spec.split(",") if p.strip()]:
        key = load_api_key_for(prov)
        if key:
            clients.append(LLMClient(api_key=key, provider=prov))
    return clients


class MultiLLMClient:
    """Try each configured client; on quota/rate errors wait and retry,
    then move to the next provider."""

    def __init__(self, clients):
        self.clients = clients or [LLMClient()]

    def _call(self, fn):
        last = None
        for i, c in enumerate(self.clients):
            try:
                return fn(c)
            except QuotaError as e:
                last = e
                if i + 1 < len(self.clients):
                    time.sleep(2)
                else:
                    time.sleep(70)
                    try:
                        return fn(c)
                    except QuotaError as e2:
                        last = e2
        raise last or RuntimeError("no LLM provider available")

    def complete(self, system, user, temperature=None, max_tokens=2048):
        return self._call(lambda c: c.complete(system, user,
                                               temperature=temperature,
                                               max_tokens=max_tokens))

    def complete_json(self, system, user, temperature=0.0, max_tokens=2048):
        return self._call(lambda c: c.complete_json(system, user,
                                                    temperature=temperature,
                                                    max_tokens=max_tokens))

    def chat_tools(self, messages, tools, temperature=0.0, max_tokens=2048):
        return self._call(lambda c: c.chat(messages, tools=tools,
                                           temperature=temperature,
                                           max_tokens=max_tokens))


class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None, provider="gemini",
                 temperature=0.2, timeout=120):
        cfg = PROVIDERS.get(provider, PROVIDERS["deepseek"])
        self.api_key = api_key or load_api_key()
        self.base_url = (base_url or os.environ.get("AVAIL_LLM_BASE_URL") or cfg["base_url"]).rstrip("/")
        self.model = model or os.environ.get("AVAIL_LLM_MODEL") or cfg["model"]
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages, temperature=None, max_tokens=2048, tools=None):
        if not self.api_key:
            raise RuntimeError(
                "No LLM API key found. Set AVAIL_LLM_API_KEY (or pass api_key=...).")
        base_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": False,
        }
        if tools:
            base_payload["tools"] = tools
            base_payload["tool_choice"] = "auto"
        for empty_try in range(2):
            payload = dict(base_payload)
            payload["max_tokens"] = (max_tokens if empty_try == 0
                                     else max(max_tokens * 2, 256))
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}",
                         "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
                                        "Gecko/20100101 Firefox/128.0")},
                method="POST",
            )
            last_err = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as e:
                    body = e.read().decode("utf-8", errors="replace")
                    bl = body.lower()
                    if e.code == 429 or (e.code == 403 and
                                         ("quota" in bl or "rate" in bl or
                                          "limit" in bl)):
                        raise QuotaError(f"LLM quota/limit ({e.code}): {body[:200]}")
                    last_err = RuntimeError(f"LLM API error {e.code}: {body[:400]}")
                    if e.code in (500, 502, 503, 504) and attempt < 2:
                        time.sleep(3 * (attempt + 1))
                        continue
                    raise last_err
            else:
                raise last_err
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            if tools:
                return msg
            content = msg.get("content") or msg.get("extra_content") or ""
            if content.strip():
                return content
            if empty_try == 0:
                time.sleep(2)
        raise RuntimeError("LLM returned an empty message (retry advised)")

    def complete(self, system, user, temperature=None, max_tokens=2048):
        return self.chat([{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                         temperature=temperature, max_tokens=max_tokens)

    def complete_json(self, system, user, temperature=0.0, max_tokens=2048):
        """Ask for a JSON object and parse it (robust to markdown fences)."""
        system += ("\nRespond with a single valid JSON object only, "
                   "no markdown fences, no extra text.")
        text = self.complete(system, user, temperature=temperature, max_tokens=max_tokens)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1] if "```" in text[3:] else text
            text = text.strip()
        # strip a leading "json" language tag if present
        if text.lower().startswith("json"):
            text = text[4:].strip()
        return json.loads(text)
