import os
import logging
import requests

logger = logging.getLogger("llm_integration")


def get_llm_response(prompt, model=None):
    """
    Get a text response from the best available LLM provider.

    Priority order:
        1. Google Gemini (free tier, 15 RPM)
        2. NVIDIA NIM (via OpenAI-compatible API)
        3. OpenRouter (free models available)

    Set the provider via env vars:
        GEMINI_API_KEY    -> enables Gemini
        NVIDIA_API_KEY    -> enables NVIDIA NIM
        OPENROUTER_API_KEY -> enables OpenRouter
    """

    # Try Gemini first
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            return _call_gemini(prompt, gemini_key, model)
        except Exception as exc:
            logger.warning("Gemini call failed, trying NVIDIA: %s", exc)

    # Try NVIDIA NIM second
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    if nvidia_key:
        try:
            return _call_nvidia(prompt, nvidia_key, model)
        except Exception as exc:
            logger.warning("NVIDIA call failed, falling back to OpenRouter: %s", exc)

    # Fallback to OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if openrouter_key:
        return _call_openrouter(prompt, openrouter_key, model)

    raise RuntimeError(
        "No LLM API key configured. Set GEMINI_API_KEY, NVIDIA_API_KEY, or OPENROUTER_API_KEY."
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


def _call_nvidia(prompt, api_key, model=None):
    """
    Call NVIDIA NIM API via OpenAI-compatible endpoint.

    NVIDIA provides an OpenAI-compatible REST API at https://integrate.api.nvidia.com/v1
    This implementation uses native requests library for consistency.
    """
    model = model or os.getenv("NVIDIA_MODEL", "minimaxai/minimax-m2.1")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "top_p": 0.95,
        "max_tokens": 512,
    }
    
    try:
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"NVIDIA API error {resp.status_code}: {resp.text[:200]}")
        
        data = resp.json()
        response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if not response_text.strip():
            raise RuntimeError("NVIDIA returned empty response")
        
        return response_text.strip()
        
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"NVIDIA API request failed: {str(exc)[:200]}")


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
