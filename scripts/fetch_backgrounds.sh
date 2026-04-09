#!/usr/bin/env bash
# fetch_backgrounds.sh — download diverse Kinetics-400 clips for CIPHER backgrounds.
# Uses yt-dlp. Targets: ~20 clips, 30-60s each, diverse scenes, minimal on-screen text.
# Run from cipher/ root: bash scripts/fetch_backgrounds.sh

set -e
OUT="data/backgrounds"
mkdir -p "$OUT"

# Format: yt-dlp picks best mp4 ≤720p, trims to 40s from a random-ish offset
download_clip() {
    local url="$1"
    local name="$2"
    local ss="${3:-5}"   # start offset in source video
    local out="$OUT/${name}.mp4"
    if [ -f "$out" ]; then echo "skip $name"; return; fi
    echo "→ $name"
    yt-dlp -f "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]" \
        --merge-output-format mp4 \
        -o "$OUT/${name}_full.%(ext)s" "$url" 2>/dev/null || { echo "  failed: $url"; return; }
    local full="$OUT/${name}_full.mp4"
    ffmpeg -y -ss "$ss" -i "$full" -t 40 -c copy "$out" 2>/dev/null
    rm -f "$full"
    echo "  saved: $out ($(du -sh "$out" | cut -f1))"
}

# ---- Diverse Kinetics / YouTube clips (no heavy text, natural scenes) ----

# Outdoor / nature
download_clip "https://www.youtube.com/watch?v=gWw23EYM9VM" "surfing_01" 10
download_clip "https://www.youtube.com/watch?v=YBDKL_QKDY0" "hiking_01" 5
download_clip "https://www.youtube.com/watch?v=pKGHXAzIvGI" "skiing_01" 15

# Sports
download_clip "https://www.youtube.com/watch?v=4NRXx6U8BNI" "basketball_01" 5
download_clip "https://www.youtube.com/watch?v=2cGMfqR_9YI" "tennis_01" 10
download_clip "https://www.youtube.com/watch?v=kFVhKi4lIBM" "soccer_01" 20

# Indoor / everyday
download_clip "https://www.youtube.com/watch?v=fHsa9DqmId8" "cooking_01" 5
download_clip "https://www.youtube.com/watch?v=zNgcYGgtf8M" "dancing_01" 8
download_clip "https://www.youtube.com/watch?v=FTQbiNvZqaY" "yoga_01" 5

# Animals / nature B-roll (common Kinetics class)
download_clip "https://www.youtube.com/watch?v=tntOCGkgt98" "animals_01" 5
download_clip "https://www.youtube.com/watch?v=nGt9jAkWie4" "birds_01" 10

# Blender open movies (clean, zero on-screen text, CC license)
download_clip "https://www.youtube.com/watch?v=YE7VzlLtp-4" "sintel_01" 60
download_clip "https://www.youtube.com/watch?v=eRsGyueVLvQ" "elephants_dream_01" 30

echo ""
echo "Done. Pool size: $(ls $OUT/*.mp4 2>/dev/null | wc -l) videos"
ls -lh "$OUT"/*.mp4 2>/dev/null || echo "(none downloaded)"
