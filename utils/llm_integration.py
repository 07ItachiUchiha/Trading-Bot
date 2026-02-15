import os
import logging
import requests

logger = logging.getLogger("llm_integration")


def get_llm_response(prompt, model=None):
    """
    Get a text response from the best available LLM provider.

    Priority order:
        1. Google Gemini (free tier, 15 RPM)
        2. OpenRouter (free models available)

    Set the provider via env vars:
        GEMINI_API_KEY   -> enables Gemini
        OPENROUTER_API_KEY -> enables OpenRouter
    """

    # Try Gemini first
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            return _call_gemini(prompt, gemini_key, model)
        except Exception as exc:
            logger.warning("Gemini call failed, falling back to OpenRouter: %s", exc)

    # Fallback to OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key:
        return _call_openrouter(prompt, openrouter_key, model)

    raise RuntimeError(
        "No LLM API key configured. Set GEMINI_API_KEY or OPENROUTER_API_KEY."
    )


def _call_gemini(prompt, api_key, model=None):
    """
    Call Google Gemini API (v1beta, generateContent).

    Free tier: 15 requests/minute, 1 million tokens/day for gemini-pro.
    """
    model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 512,
        },
    }

    resp = requests.post(url, json=payload, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    if not text.strip():
        raise RuntimeError("Gemini returned empty response")

    return text.strip()


def _call_openrouter(prompt, api_key, model=None):
    """
    Call OpenRouter unified API.

    Free models like meta-llama/llama-4-maverick:free are available.
    """
    model = model or os.getenv("LLM_MODEL", "meta-llama/llama-4-maverick:free")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=15,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text[:200]}")

    return resp.json()["choices"][0]["message"]["content"]
