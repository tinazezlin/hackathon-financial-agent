"""Model access (MOCK of the hackathon's src/core/llm.py).

At the hackathon: get_llm("<model id>") returns a STACKIT model via a local
proxy and you must NOT edit this file. Here we emulate it with:

    LLM_PROVIDER=groq    (default) Groq API, needs GROQ_API_KEY. Runs the same
                         gpt-oss models the hackathon offers.
    LLM_PROVIDER=ollama  local Ollama (OLLAMA_MODEL, default llama3.1:8b)
    LLM_PROVIDER=openai  any OpenAI-compatible endpoint (OPENAI_BASE_URL, OPENAI_API_KEY)
    LLM_PROVIDER=mock    no LLM at all: agents fall back to deterministic
                         heuristics. Free, fast, good for testing the plumbing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# Hackathon model ids -> closest stand-in on Groq
GROQ_MODEL_MAP = {
    "openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "Qwen/Qwen3.8-27B": "qwen/qwen3-32b",
    "Qwen/Qwen3.6-27B": "qwen/qwen3-32b",
    "cortecs/Llama-3.3-70B-Instruct-FP8-Dynamic": "llama-3.3-70b-versatile",
    "google/gemma-4-31": "openai/gpt-oss-120b",
}


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").lower()


def is_mock() -> bool:
    return provider() == "mock"


@lru_cache(maxsize=16)
def get_llm(model_id: str = "openai/gpt-oss-120b", temperature: float = 0.0):
    """Return a LangChain chat model, or None in mock mode."""
    p = provider()
    if p == "mock":
        return None
    if p == "groq":
        from langchain_groq import ChatGroq

        model = GROQ_MODEL_MAP.get(model_id, model_id)
        extra = {}
        if "gpt-oss" in model:
            # Reasoning tokens count against max_tokens: with the default budget gpt-oss can
            # spend everything on reasoning and return EMPTY content (finish_reason="length").
            extra["reasoning_effort"] = os.getenv("REASONING_EFFORT", "medium")
        return ChatGroq(model=model, temperature=temperature, max_retries=8,
                        max_tokens=int(os.getenv("MAX_TOKENS", "16000")), **extra)
    if p == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=temperature)
    if p == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_id, temperature=temperature, base_url=os.environ["OPENAI_BASE_URL"],
                          api_key=os.environ.get("OPENAI_API_KEY", "none"), max_retries=8)
    raise ValueError(f"Unknown LLM_PROVIDER={p!r}")
