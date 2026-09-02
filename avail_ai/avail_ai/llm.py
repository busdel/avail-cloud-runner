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
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    # Google Gemini via its OpenAI-compatible endpoint (no SDK needed).
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "model": "gemini-3.6-flash"},
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


class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None, provider="gemini",
                 temperature=0.2, timeout=120):
        cfg = PROVIDERS.get(provider, PROVIDERS["deepseek"])
        self.api_key = api_key or load_api_key()
        self.base_url = (base_url or os.environ.get("AVAIL_LLM_BASE_URL") or cfg["base_url"]).rstrip("/")
        self.model = model or os.environ.get("AVAIL_LLM_MODEL") or cfg["model"]
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages, temperature=None, max_tokens=2048):
        if not self.api_key:
            raise RuntimeError(
                "No LLM API key found. Set AVAIL_LLM_API_KEY (or pass api_key=...).")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
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
                last_err = RuntimeError(f"LLM API error {e.code}: {body[:400]}")
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise last_err
        else:
            raise last_err
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or msg.get("extra_content") or ""
        if not content.strip():
            raise RuntimeError("LLM returned an empty message (retry advised)")
        return content

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
