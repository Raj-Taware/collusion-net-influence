"""Gemini client with an on-disk response cache.

Every call is keyed by (model, system, prompt, temperature, seed). Re-running an experiment with
the same seeds replays cached responses, so results are reproducible and reruns cost $0.
Commit `cache/` alongside results if you want others to reproduce figures without an API key.

Needs `pip install google-genai` and GEMINI_API_KEY in the environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class CachedGemini:
    model: str = field(default_factory=lambda: os.environ.get("NETINF_MODEL", ""))
    cache_path: Path = Path("cache/llm.sqlite")
    max_retries: int = 5
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self):
        if not self.model:
            raise ValueError("set a model name, e.g. CachedGemini(model=...) or NETINF_MODEL env var")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.cache_path)
        self._db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, request TEXT, response TEXT)")
        self._client = None

    def generate(self, prompt: str, system: str = "", temperature: float = 1.0, seed: int = 0,
                 max_output_tokens: int = 512) -> str:
        request = {"model": self.model, "system": system, "prompt": prompt, "temperature": temperature,
                   "seed": seed, "max_output_tokens": max_output_tokens}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        row = self._db.execute("SELECT response FROM calls WHERE key = ?", (key,)).fetchone()
        if row:
            self.usage.cache_hits += 1
            return row[0]

        text = self._call(request)
        self._db.execute("INSERT INTO calls VALUES (?, ?, ?)", (key, json.dumps(request), text))
        self._db.commit()
        return text

    def _call(self, req: dict) -> str:
        if self._client is None:
            from google import genai

            self._client = genai.Client()  # reads GEMINI_API_KEY
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=req["system"] or None,
            temperature=req["temperature"],
            seed=req["seed"],
            max_output_tokens=req["max_output_tokens"],
        )
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(model=req["model"], contents=req["prompt"], config=config)
                break
            except Exception:  # rate limits / transient errors: back off and retry
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        self.usage.calls += 1
        if resp.usage_metadata:
            self.usage.input_tokens += resp.usage_metadata.prompt_token_count or 0
            self.usage.output_tokens += resp.usage_metadata.candidates_token_count or 0
        return resp.text or ""
