"""Which LLM providers are usable right now, detected from .env API keys.

Add a key for any of these in .env (e.g. OPENAI_API_KEY=...) and it shows up
automatically as a comparable provider in the app — no code changes needed.
"""
import os

# provider -> (display label, API key env var, default quick model, default deep model)
PROVIDER_REGISTRY = {
    "google":    ("Gemini",    "GOOGLE_API_KEY",    "gemini-3.5-flash-lite", "gemini-3.8-flash"),
    "openai":    ("GPT",       "OPENAI_API_KEY",    "gpt-6-luna",            "gpt-6-sol"),
    "anthropic": ("Claude",    "ANTHROPIC_API_KEY", "claude-haiku-4-5",      "claude-sonnet-5"),
    "xai":       ("Grok",      "XAI_API_KEY",       "grok-4.6",              "grok-4.6"),
    "deepseek":  ("DeepSeek",  "DEEPSEEK_API_KEY",  "deepseek-flash",        "deepseek-v4-pro"),
    "mistral":   ("Mistral",   "MISTRAL_API_KEY",   "mistral-small-2603",    "mistral-medium-2604"),
}


def list_providers() -> list[dict]:
    return [
        {
            "provider": key,
            "label": label,
            "available": bool(os.environ.get(env_var)),
        }
        for key, (label, env_var, _, _) in PROVIDER_REGISTRY.items()
    ]


def available_providers() -> list[str]:
    return [key for key, (_, env_var, _, _) in PROVIDER_REGISTRY.items() if os.environ.get(env_var)]


def provider_models(provider: str) -> tuple[str, str]:
    """(quick_model, deep_model) defaults for a provider."""
    _, _, quick, deep = PROVIDER_REGISTRY[provider]
    return quick, deep


def provider_label(provider: str) -> str:
    return PROVIDER_REGISTRY.get(provider, (provider, None, None, None))[0]
