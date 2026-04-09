"""
vlm_backends.py — VLM function factories for CIPHER evaluator.

Each backend returns a vlm_fn(prompt, images=None) -> str
compatible with solve_captcha.py's make_vlm_fn interface.

Backends:
  openai_compat(endpoint, model, api_key)  — vllm / lm-studio / OpenAI
  anthropic(model)                          — Claude via Anthropic SDK
  swarm_codex(effort)                       — GPT-5.x via Swarm MCP (stub)
  swarm_gemini(effort)                      — Gemini via Swarm MCP (stub)
"""
import base64
import io
import os
import sys


# ---------------------------------------------------------------------------
# OpenAI-compatible (vllm, LM Studio, OpenAI API)
# ---------------------------------------------------------------------------

def openai_compat(endpoint: str, model: str, api_key: str = "EMPTY"):
    """Works with any OpenAI-compatible endpoint (vllm, lm-studio, openai)."""
    from openai import OpenAI
    client = OpenAI(base_url=endpoint, api_key=api_key)

    def vlm_fn(prompt: str, images=None) -> str:
        content = [{"type": "text", "text": prompt}]
        if images:
            from PIL import Image
            for img in images[:8]:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=256,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    vlm_fn.__name__ = f"openai/{model}"
    return vlm_fn


# ---------------------------------------------------------------------------
# Anthropic (Claude Haiku / Sonnet / Opus)
# ---------------------------------------------------------------------------

def anthropic_claude(model: str = "claude-haiku-4-5-20251001"):
    """Claude via Anthropic SDK. Uses ANTHROPIC_API_KEY env var."""
    import anthropic as ant

    client = ant.Anthropic()

    def vlm_fn(prompt: str, images=None) -> str:
        content = []
        if images:
            from PIL import Image
            for img in images[:8]:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })
        content.append({"type": "text", "text": prompt})
        msg = client.messages.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": content}],
        )
        return msg.content[0].text.strip()

    vlm_fn.__name__ = f"anthropic/{model}"
    return vlm_fn


# ---------------------------------------------------------------------------
# Google Gemini (via google-generativeai)
# ---------------------------------------------------------------------------

def google_gemini(model: str = "gemini-2.0-flash"):
    """Gemini via google-generativeai SDK. Uses GOOGLE_API_KEY env var."""
    import google.generativeai as genai
    from PIL import Image as PILImage

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gmodel = genai.GenerativeModel(model)

    def vlm_fn(prompt: str, images=None) -> str:
        parts = []
        if images:
            for img in images[:8]:
                parts.append(img)
        parts.append(prompt)
        resp = gmodel.generate_content(parts)
        return resp.text.strip()

    vlm_fn.__name__ = f"gemini/{model}"
    return vlm_fn


# ---------------------------------------------------------------------------
# GPT-4o / GPT-5 via OpenAI API
# ---------------------------------------------------------------------------

def openai_gpt(model: str = "gpt-4o-mini"):
    """OpenAI GPT models. Uses OPENAI_API_KEY env var."""
    return openai_compat(
        endpoint="https://api.openai.com/v1",
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )


# ---------------------------------------------------------------------------
# Backend registry — used by run_cipher.py CLI
# ---------------------------------------------------------------------------

REGISTRY = {
    # Local / self-hosted
    "qwen2-vl-7b":    lambda: openai_compat("http://localhost:8000/v1", "qwen2-vl-7b-instruct"),
    "qwen2-vl-72b":   lambda: openai_compat("http://localhost:8000/v1", "qwen2-vl-72b-instruct"),
    "internvl2-8b":   lambda: openai_compat("http://localhost:8000/v1", "internvl2-8b"),
    # API
    "haiku":          lambda: anthropic_claude("claude-haiku-4-5-20251001"),
    "sonnet":         lambda: anthropic_claude("claude-sonnet-4-6"),
    "gpt-4o-mini":    lambda: openai_gpt("gpt-4o-mini"),
    "gpt-4o":         lambda: openai_gpt("gpt-4o"),
    "gemini-flash":   lambda: google_gemini("gemini-2.0-flash"),
    "gemini-pro":     lambda: google_gemini("gemini-1.5-pro"),
    # OCR-only baseline (no VLM)
    "ocr-only":       lambda: None,
}


def get_backend(name: str):
    if name not in REGISTRY:
        raise ValueError(f"Unknown backend {name!r}. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
