import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Genuinely free (no per-token cost) OpenRouter model, verified against the
# live https://openrouter.ai/api/v1/models catalog on 2026-09-05. Free-tier
# OpenRouter requests are rate-limited (20/min, 50/day with no credits).
OPENROUTER_FREE_MODEL = "z-ai/glm-5.2:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_chat_llm(**kwargs) -> ChatOpenAI:
    """Local dev (no OPENROUTER_API_KEY): gpt-4o-mini via OPENAI_API_KEY, unchanged.
    Production (OPENROUTER_API_KEY set): routed through OpenRouter's free model so
    public demo traffic never touches Sahil's OpenAI billing."""
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return ChatOpenAI(
            model=OPENROUTER_FREE_MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=openrouter_key,
            **kwargs,
        )
    if os.environ.get("VERCEL"):
        # Never let deployed traffic fall back to direct OpenAI billing.
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Refusing to fall back to direct "
            "OpenAI billing on the deployed instance - set OPENROUTER_API_KEY "
            "in the Vercel project's environment variables."
        )
    return ChatOpenAI(model="gpt-4o-mini", **kwargs)
