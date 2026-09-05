import json
import os
import re
from typing import TypeVar

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

# Genuinely free (no per-token cost) OpenRouter models, primary first.
# Measured on 2026-09-05 against a realistic 6k-char grading prompt: the two
# Google endpoints and glm-5.2 all returned 429 from the shared free pool,
# minimax-m3 answered in 1.5s with clean JSON, and the nemotron models
# answered but prefixed their reasoning. Provider-level fallback is not
# enough on its own, because a model whose only free provider is saturated
# has nowhere else to go, so the fallback has to span models too.
OPENROUTER_FREE_MODELS = [
    "minimax/minimax-m3:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "z-ai/glm-5.2:free",
]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_chat_llm(**kwargs) -> ChatOpenAI:
    """Local dev (no OPENROUTER_API_KEY): gpt-4o-mini via OPENAI_API_KEY, unchanged.
    Production (OPENROUTER_API_KEY set): routed through OpenRouter's free model so
    public demo traffic never touches Sahil's OpenAI billing."""
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return ChatOpenAI(
            model=OPENROUTER_FREE_MODELS[0],
            base_url=OPENROUTER_BASE_URL,
            api_key=openrouter_key,
            max_retries=3,
            timeout=45,
            extra_body={
                # `models` is OpenRouter's own cross-model fallback: a 429 on
                # the primary moves to the next id in one request, no retry
                # round trip. `provider` fallback only helps within a model.
                "models": OPENROUTER_FREE_MODELS[1:],
                "provider": {"allow_fallbacks": True, "sort": "throughput"},
            },
            **kwargs,
        )
    if os.environ.get("VERCEL") or os.environ.get("RENDER"):
        # Never let deployed traffic fall back to direct OpenAI billing.
        # Checked both: Render auto-sets RENDER=true, Vercel sets VERCEL=1 -
        # relying on only one of these previously meant this check silently
        # did nothing on whichever host wasn't explicitly named.
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Refusing to fall back to direct "
            "OpenAI billing on the deployed instance - set OPENROUTER_API_KEY "
            "in the host's environment variables."
        )
    return ChatOpenAI(model="gpt-4o-mini", **kwargs)


T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def invoke_structured(llm: ChatOpenAI, schema: type[T], prompt: str) -> tuple[T | None, str]:
    """Ask for `schema` as JSON and parse it, returning (parsed_or_None, raw_text).

    with_structured_output raises on a model that answers in prose. Open-weight
    free models do that often enough that a hard failure means a 500 for the
    user, so the caller gets the raw text back and decides how to degrade.
    """
    instruction = (
        f"{prompt}\n\n"
        "Respond with ONLY a JSON object matching this schema, no prose, no "
        f"code fences:\n{json.dumps(schema.model_json_schema())}"
    )
    raw = str(llm.invoke(instruction).content).strip()

    candidates = [raw]
    fenced = _JSON_BLOCK.search(raw)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            return schema.model_validate_json(candidate), raw
        except (ValidationError, ValueError):
            continue
    return None, raw
