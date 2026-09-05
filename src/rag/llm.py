import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Genuinely free (no per-token cost) OpenRouter model. Checked provider
# redundancy via /api/v1/models/.../endpoints on 2026-09-05: this one has 9
# providers serving it (Google, Cloudflare, DeepInfra, ...) vs. z-ai/glm-5.2's
# single free-tier provider (Decart), which we saw get congested under load.
# No free model is ever guaranteed available - OpenRouter's own account-level
# limit is still 20 req/min, 50/day with no credits added.
OPENROUTER_FREE_MODEL = "google/gemma-4-26b-a4b-it:free"
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
            # Free-tier providers get congested (seen it happen on two
            # different models/providers in one session). By default
            # OpenRouter tries one provider per request and fails if that
            # one is down - explicit fallback routing lets it retry a
            # *different* free provider for the same model within one call,
            # which is what its own 429 error message recommends.
            max_retries=5,
            timeout=30,
            extra_body={"provider": {"allow_fallbacks": True, "sort": "throughput"}},
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
