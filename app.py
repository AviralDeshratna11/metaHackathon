"""
app.py — HuggingFace Space entry point
Serves the Pokemon Red RL environment via a FastAPI HTTP server.
Required by hackathon: must return 200 on ping and respond to /reset().
"""

import os
import json
import time
import asyncio
from typing import Any, Dict, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

import sys
sys.path.insert(0, os.path.dirname(__file__))

from envs.pokemon_red_env import PokemonRedStubEnv

app = FastAPI(
    title="Pokemon Red RL Environment",
    description="Mini-RL environment for Pokemon Red — hackathon submission",
    version="1.0.0",
)

# ── Global env instance ───────────────────────────────────────────────────
_env: Optional[PokemonRedStubEnv] = None
_current_obs: Optional[Dict]       = None


def get_env() -> PokemonRedStubEnv:
    global _env
    if _env is None:
        _env = PokemonRedStubEnv(max_steps=int(os.environ.get("MAX_STEPS", "256")))
    return _env


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Health check — hackathon pings this."""
    return JSONResponse({"status": "ok", "env": "PokemonRed-v1", "time": time.time()})


@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})


@app.post("/reset")
async def reset(task: str = "obtain_first_badge"):
    """Reset the environment and return first observation."""
    global _current_obs
    env = get_env()
    env.task = task
    obs = env.reset()
    _current_obs = {
        "screen_shape": list(obs["screen"].shape),
        "game_state":   obs["game_state"].tolist(),
    }
    return JSONResponse({
        "status":    "reset",
        "task":      task,
        "obs_keys":  list(obs.keys()),
        "game_state": obs["game_state"].tolist(),
    })


@app.post("/step")
async def step(action: int = 0):
    """Execute one step."""
    global _current_obs
    env = get_env()
    if action not in range(8):
        raise HTTPException(status_code=400, detail=f"Invalid action {action}")
    obs, reward, done, info = env.step(action)
    _current_obs = {"game_state": obs["game_state"].tolist()}
    return JSONResponse({
        "reward":     float(reward),
        "done":       bool(done),
        "info":       info,
        "game_state": obs["game_state"].tolist(),
    })


@app.get("/state")
async def state():
    """Return current environment state."""
    env = get_env()
    return JSONResponse(env.state())


@app.get("/tasks")
async def tasks():
    """List all available tasks."""
    from graders.task_graders import GRADER_REGISTRY
    return JSONResponse({
        "tasks": [
            {"id": tid, "description": g.description}
            for tid, g in GRADER_REGISTRY.items()
        ]
    })


@app.get("/metadata")
async def metadata():
    """Return env metadata (OpenEnv spec)."""
    from envs.pokemon_red_env import PokemonRedEnv
    return JSONResponse(PokemonRedEnv.metadata)


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
