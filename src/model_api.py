from __future__ import annotations
import os, json, requests

# --- Qwen (DashScope / Model Studio) ---
DS_API_KEY  = os.getenv("DASHSCOPE_API_KEY", "")
DS_API_BASE = os.getenv("DASHSCOPE_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
DS_MODEL    = os.getenv("DASHSCOPE_MODEL", "qwen-flash")   # e.g., qwen-flash / qwen-plus / qwen-max

# --- Kimi (Moonshot) fallback (optional) ---
MS_API_KEY  = os.getenv("MOONSHOT_API_KEY", "")
MS_API_BASE = os.getenv("MOONSHOT_API_BASE", "https://api.moonshot.ai/v1")
MS_MODEL    = os.getenv("MOONSHOT_MODEL", "kimi-k2-0905-preview")

# --- OpenAI-compatible fallback (optional) ---
OA_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OA_API_BASE = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
OA_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TIMEOUT = float(os.getenv("TIMEOUT", "60"))

def _post_chat(base: str, api_key: str, model: str, prompt: str) -> str:
    """
    Generic OpenAI-compatible /chat/completions call.
    Works for DashScope (Qwen), Moonshot (Kimi), OpenAI, etc.
    """
    url = f"{base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful model that follows instructions exactly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 8,  # only need a single letter like "(C)"
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

def call_model(prompt: str) -> str:
    # Prefer Qwen (DashScope)
    if DS_API_KEY:
        return _post_chat(DS_API_BASE, DS_API_KEY, DS_MODEL, prompt)
    # Then Kimi (Moonshot)
    if MS_API_KEY:
        return _post_chat(MS_API_BASE, MS_API_KEY, MS_MODEL, prompt)
    # Then OpenAI-compatible
    if OA_API_KEY:
        return _post_chat(OA_API_BASE, OA_API_KEY, OA_MODEL, prompt)
    raise RuntimeError(
        "No API key configured. Set DASHSCOPE_API_KEY for Qwen (recommended), "
        "or MOONSHOT_API_KEY (Kimi), or OPENAI_API_KEY."
    )