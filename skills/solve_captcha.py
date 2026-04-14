"""
solve_captcha.py — main agent loop for video-captcha-solver.

Usage:
    python solve_captcha.py --video input.mp4 \
        --instructions "cut seconds 3-7, reverse, flip horizontally" \
        [--vlm-endpoint http://localhost:8000/v1] \
        [--vlm-model qwen2-vl-7b-instruct] \
        [--max-retries 3]

Returns the key string to stdout (exit 0) or exits 1 if not found.
"""
import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path

# local imports
sys.path.insert(0, str(Path(__file__).parent))
from edit_ops import parse_instructions, build_ffmpeg_commands
from extract_key import extract_key


# ---------------------------------------------------------------------------
# VLM client (OpenAI-compatible, works with vllm / lm-studio / openai)
# ---------------------------------------------------------------------------

def make_vlm_fn(endpoint: str, model: str, api_key: str = "EMPTY"):
    try:
        from openai import OpenAI
        import base64, io
    except ImportError:
        print("[warn] openai not installed — VLM extraction disabled", file=sys.stderr)
        return None

    client = OpenAI(base_url=endpoint, api_key=api_key)

    def vlm_fn(prompt: str, images=None) -> str:
        content = [{"type": "text", "text": prompt}]
        if images:
            from PIL import Image
            import base64, io
            for img in images[:8]:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=256,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    return vlm_fn


# ---------------------------------------------------------------------------
# Core solve loop
# ---------------------------------------------------------------------------

def solve(video_path: str, instructions: str, vlm_fn=None, max_retries: int = 3) -> str | None:
    for attempt in range(max_retries):
        print(f"[solve] attempt {attempt + 1}/{max_retries}", file=sys.stderr)

        # 1. Parse instructions (retry with VLM if regex fails)
        ops = parse_instructions(instructions, vlm_fn=vlm_fn if attempt > 0 else None)
        if not ops:
            print("[solve] no ops parsed", file=sys.stderr)
            if attempt == max_retries - 1:
                break
            continue

        print(f"[solve] ops: {ops}", file=sys.stderr)

        # 2. Build and run ffmpeg chain
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "output.mp4")
            cmds, tmp_files = build_ffmpeg_commands(video_path, ops, out_path)

            success = True
            for cmd in cmds:
                print(f"[solve] running: {' '.join(cmd)}", file=sys.stderr)
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    print(f"[solve] ffmpeg failed: {result.stderr.decode()[-500:]}", file=sys.stderr)
                    success = False
                    break

            if not success or not Path(out_path).exists():
                continue

            if Path(out_path).stat().st_size < 1000:
                print("[solve] output too small, likely empty", file=sys.stderr)
                continue

            # 3. Extract key
            key = extract_key(out_path, vlm_fn=vlm_fn)
            if key:
                return key

        print(f"[solve] no key found on attempt {attempt + 1}, retrying with swapped op order...",
              file=sys.stderr)
        # Retry: try reversing op order (common captcha trick)
        if attempt == 1:
            ops = list(reversed(ops))

    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Solve video-editing captcha")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--instructions", required=True, help="NL editing instructions")
    parser.add_argument("--vlm-endpoint", default="http://localhost:8000/v1",
                        help="OpenAI-compatible VLM endpoint")
    parser.add_argument("--vlm-model", default="qwen2-vl-7b-instruct",
                        help="VLM model name")
    parser.add_argument("--vlm-api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--no-vlm", action="store_true",
                        help="Disable VLM, use only QR/OCR/audio/LSB")
    args = parser.parse_args()

    vlm_fn = None
    if not args.no_vlm:
        vlm_fn = make_vlm_fn(args.vlm_endpoint, args.vlm_model, args.vlm_api_key)

    key = solve(
        video_path=args.video,
        instructions=args.instructions,
        vlm_fn=vlm_fn,
        max_retries=args.max_retries,
    )

    if key:
        print(key)
        sys.exit(0)
    else:
        print("NOT_FOUND", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
