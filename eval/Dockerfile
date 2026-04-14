FROM python:3.12-slim

# System deps: ffmpeg + zbar (for pyzbar QR decoding)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libzbar0 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /cipher

# Python deps (cached layer)
COPY eval/requirements.txt /cipher/eval/requirements.txt
RUN pip install --no-cache-dir -r eval/requirements.txt

# Source code
COPY src/ /cipher/src/
COPY run_cipher.py /cipher/

# Skill scripts (edit_ops, extract_key, solve_captcha)
COPY skills/ /cipher/skills/

# Patch evaluator to find skills in /cipher/skills instead of ~/.hermes
ENV CIPHER_SKILL_DIR=/cipher/skills

# Data and output mount points
VOLUME ["/data", "/runs"]

ENTRYPOINT ["python", "run_cipher.py", "bench"]
CMD ["--manifest", "/data/manifest.json", "--model", "ocr-only", "--runs-dir", "/runs"]
