# ── Pokemon Red RL Environment — Hackathon Docker Image ──────────────────
FROM python:3.11-slim

LABEL maintainer="pokemon-rl-hackathon"
LABEL description="Mini-RL environment for Pokemon Red with PPO agent"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsdl2-dev \
        libsdl2-image-dev \
        libsdl2-mixer-dev \
        ffmpeg \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p saves checkpoints logs

# Env var defaults (overridden at runtime by hackathon platform)
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV ROM_PATH="pokemon_red.gb"
ENV USE_STUB="1"
ENV MAX_STEPS="256"
ENV CHECKPOINT="checkpoints/best.pt"
ENV PYTHONUNBUFFERED="1"

# Health-check: verify inference.py exists and imports cleanly
RUN python -c "import sys; sys.path.insert(0,''); from envs.pokemon_red_env import PokemonRedStubEnv; print('OK')"

EXPOSE 7860

# Default: run inference script (hackathon evaluation entry point)
CMD ["python", "inference.py", "--stub", "--max_steps", "128"]
