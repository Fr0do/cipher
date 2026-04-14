#!/usr/bin/env python3
"""
eval_yellow_ocr.py — OCR baseline that isolates YELLOW text before reading.

Waits for calibrate_overnight.py to produce results.json, then:
1. For each sample, loads corrected frames
2. Masks to yellow-ish pixels (HSV hue ~25-35, high saturation)
3. Runs easyocr/tesseract on masked regions
4. Reports EM/CF1

This is a smarter OCR baseline that uses color heuristics — no VLM needed.

Usage:
    python scripts/eval_yellow_ocr.py --results /tmp/cipher_overnight/results.json
"""
import sys, os, json, time, re
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evaluator import exact_match, char_f1


def isolate_yellow(img_array: np.ndarray) -> np.ndarray:
    """Mask yellow-ish pixels. Returns binary mask."""
    import cv2
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    # Yellow in HSV: hue ~20-40, sat > 100, val > 150
    lower = np.array([15, 80, 150])
    upper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # Dilate to connect nearby text pixels
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def extract_yellow_text(frame_paths: list[str]) -> str:
    """Extract text from yellow regions of frames."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], verbose=False)
    except ImportError:
        return ""

    import cv2
    candidates = []

    for fp in frame_paths:
        try:
            img = np.array(Image.open(fp).convert("RGB"))
        except Exception:
            continue

        mask = isolate_yellow(img)

        # Check if any yellow pixels exist
        if mask.sum() < 100:
            continue

        # Apply mask: keep yellow region, white background
        masked = np.ones_like(img) * 255
        masked[mask > 0] = img[mask > 0]

        # Also try: invert for better OCR (dark text on white)
        # But our text is bright yellow, so let's make it black on white
        gray_masked = cv2.cvtColor(masked.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray_masked, 200, 255, cv2.THRESH_BINARY_INV)

        # Run OCR on binary mask
        results = reader.readtext(binary)
        for _, text, conf in results:
            text = text.strip().upper()
            # Filter: 5-8 chars, alphanumeric, no forbidden chars
            text = re.sub(r'[^A-Z0-9]', '', text)
            if conf > 0.3 and 5 <= len(text) <= 9:
                candidates.append((conf, text))

        # Also try OCR on original cropped region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 30 and h > 10:  # reasonable text bbox
                crop = img[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
                results2 = reader.readtext(crop)
                for _, text, conf in results2:
                    text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                    if conf > 0.3 and 5 <= len(text) <= 9:
                        candidates.append((conf + 0.1, text))  # bonus for cropped

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return ""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="/tmp/cipher_overnight/results.json")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds to wait if results not ready")
    parser.add_argument("--max-wait", type=int, default=3600, help="Max seconds to wait for results")
    args = parser.parse_args()

    # Wait for results file
    waited = 0
    while not os.path.exists(args.results):
        if waited >= args.max_wait:
            print(f"Timeout waiting for {args.results}")
            sys.exit(1)
        print(f"Waiting for {args.results}... ({waited}s)")
        time.sleep(args.poll_interval)
        waited += args.poll_interval

    print("Loading results and evaluating with yellow-OCR baseline...")

    with open(args.results) as f:
        results = json.load(f)

    eval_results = {}

    for level, records in results.items():
        level_results = []
        print(f"\n=== {level} ===")

        for r in records:
            corrected_frames = r.get("corrected_frames", [])
            if not corrected_frames:
                print(f"  - {r['sample_id']}: no corrected frames, skip")
                level_results.append({"sample_id": r["sample_id"], "pred": "", "em": False, "cf1": 0.0})
                continue

            pred = extract_yellow_text(corrected_frames)
            em = exact_match(pred, r["gold_key"])
            cf1_score = char_f1(pred, r["gold_key"])

            mark = "✓" if em else ("~" if cf1_score > 0.5 else "✗")
            print(f"  {mark} {r['sample_id']} gold={r['gold_key']} pred={pred!r:10s} "
                  f"em={em} cf1={cf1_score:.2f}")

            level_results.append({
                "sample_id": r["sample_id"],
                "pred": pred,
                "em": em,
                "cf1": round(cf1_score, 3),
            })

        eval_results[level] = level_results

    # Summary
    print("\n" + "=" * 50)
    print(f"{'Level':<6} {'N':>3} {'Yellow-OCR EM':>14} {'CF1':>8}")
    print("-" * 35)
    for level, recs in eval_results.items():
        if not recs:
            continue
        n = len(recs)
        em = sum(r["em"] for r in recs) / n
        cf1_avg = sum(r["cf1"] for r in recs) / n
        print(f"{level:<6} {n:>3} {em:>14.0%} {cf1_avg:>8.3f}")

    out_path = args.results.replace("results.json", "yellow_ocr_results.json")
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
