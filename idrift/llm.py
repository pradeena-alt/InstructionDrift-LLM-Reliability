"""Model backends.

  mock           offline simulator for TESTING THE PIPELINE ONLY (results are not scientific)
  openai_compat  any OpenAI-compatible /chat/completions server (OpenAI, Ollama, vLLM, LM Studio, ...)
  anthropic      Anthropic Messages API

All real calls are cached on disk (sqlite) keyed by (backend, model, messages, params), so
re-running an experiment is free and reproducible.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
import sqlite3
import threading
import time

from .constraints import render_response


class Cache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            r = self.conn.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return r[0] if r else None

    def put(self, k, v):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, v))
            self.conn.commit()


class BaseLLM:
    name = "base"

    def chat(self, messages: list, meta: dict | None = None) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------- mock
class MockLLM(BaseLLM):
    """Simulated model whose compliance decays with turns since a rule was last stated.

    It reads `meta` (ground-truth rule state) which real backends never see. Use it to
    verify that the pipeline, verifiers, policies, risk model and analysis all work.
    """
    name = "mock"
    BASE_P = 0.995
    RATE = {"format": 0.03, "length": 0.05, "negative": 0.09, "terminology": 0.07, "persistence": 0.08}

    def __init__(self, cfg: dict):
        self.rate_scale = float(cfg.get("mock_rate_scale", 1.0))

    def chat(self, messages, meta=None):
        meta = meta or {}
        seed = hashlib.sha256(json.dumps(messages).encode()).hexdigest()
        rng = random.Random(seed)
        base = meta.get("base_text", "Sure, happy to chat about that.")
        mult = 1.0 + 0.8 * meta.get("off_frac3", 0.0) + 0.3 * meta.get("topic_shift", 0)
        flags = {}
        for c in meta.get("active", []):
            ts = meta["turns_since"][c.cid]
            p = self.BASE_P * math.exp(-self.RATE[c.cls] * self.rate_scale * ts * mult)
            flags[c] = rng.random() < p
        return render_response(base, flags)


# --------------------------------------------------------------------------- real backends
class _HTTPLLM(BaseLLM):
    def __init__(self, cfg: dict):
        import requests  # lazy import
        self.requests = requests
        self.model = cfg["model"]
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 300)
        self.sleep = cfg.get("request_sleep", 0.0)
        self.retries = cfg.get("retries", 5)
        self.cache = Cache(cfg.get("cache_path", "results/cache.sqlite"))
        self.key = os.environ.get(cfg.get("api_key_env", ""), "") if cfg.get("api_key_env") else ""

    def _key(self, messages):
        blob = json.dumps([self.name, self.model, self.temperature, self.max_tokens, messages])
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages, meta=None):
        k = self._key(messages)
        hit = self.cache.get(k)
        if hit is not None:
            return hit
        last = None
        for attempt in range(self.retries):
            try:
                out = self._call(messages)
                self.cache.put(k, out)
                if self.sleep:
                    time.sleep(self.sleep)
                return out
            except Exception as e:  # network / rate-limit
                last = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"LLM call failed after retries: {last}")

    def _call(self, messages):
        raise NotImplementedError


class OpenAICompatLLM(_HTTPLLM):
    name = "openai_compat"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.base_url = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")

    def _call(self, messages):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        r = self.requests.post(f"{self.base_url}/chat/completions", headers=h, timeout=120, json={
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": self.max_tokens})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


class AnthropicLLM(_HTTPLLM):
    name = "anthropic"

    def _call(self, messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        r = self.requests.post("https://api.anthropic.com/v1/messages", timeout=120, headers={
            "x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                  "system": system, "messages": msgs})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])


def make_llm(cfg: dict) -> BaseLLM:
    b = cfg.get("backend", "mock")
    if b == "mock":
        return MockLLM(cfg)
    if b == "openai_compat":
        return OpenAICompatLLM(cfg)
    if b == "anthropic":
        return AnthropicLLM(cfg)
    raise ValueError(f"unknown backend {b}")
