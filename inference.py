#!/usr/bin/env python3
"""
inference.py — Pokemon Red RL Agent Inference Script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hackathon required: structured [START] / [STEP] / [END] stdout logs.
Uses OpenAI-compatible client (API_BASE_URL + MODEL_NAME env vars).
Runs the agent through all tasks and produces graded scores in 0-1.
"""

import os
import sys
import json
import time
import math
import argparse
import traceback
from typing import Any, Dict, List

# ── Env vars (hackathon mandated) ─────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ROM_PATH     = os.environ.get("ROM_PATH",     "pokemon_red.gb")
CHECKPOINT   = os.environ.get("CHECKPOINT",   "checkpoints/best.pt")
USE_STUB     = os.environ.get("USE_STUB",     "1") == "1"   # set 0 for real ROM
MAX_STEPS    = int(os.environ.get("MAX_STEPS", "256"))

# ── OpenAI client (mandatory per spec) ────────────────────────────────────
try:
    from openai import OpenAI
    openai_client = OpenAI(
        api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", "dummy"),
        base_url=API_BASE_URL,
    )
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    openai_client   = None

# ── Local imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from envs.pokemon_red_env import PokemonRedStubEnv, PokemonRedEnv
from graders.task_graders import GRADER_REGISTRY, run_all_graders

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── LLM helper (OpenAI client per spec) ───────────────────────────────────
def llm_action_hint(state: Dict, task: str) -> int:
    """
    Query the LLM for a high-level hint; translate to discrete action.
    Falls back to random if LLM unavailable or errors out.
    """
    if not OPENAI_AVAILABLE or not openai_client:
        return _random_action()

    try:
        badges   = state.get("badges", 0)
        level    = state.get("level",  1)
        maps     = state.get("maps_visited", 0)

        prompt = (
            f"You are advising a Pokemon Red RL agent.\n"
            f"Task: {task}\n"
            f"State: badges={badges}, level={level}, maps_visited={maps}\n"
            f"Choose ONE action from: 0=down 1=left 2=right 3=up 4=A 5=B 6=start 7=select\n"
            f"Respond with ONLY the integer action number."
        )
        resp = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        action = int(text)
        if 0 <= action <= 7:
            return action
    except Exception:
        pass
    return _random_action()


def _random_action() -> int:
    import random
    return random.randint(0, 7)


# ── Policy (PPO checkpoint or random fallback) ────────────────────────────
class Policy:
    def __init__(self, checkpoint_path: str = ""):
        self.model = None
        self.device = None
        if TORCH_AVAILABLE and checkpoint_path and os.path.exists(checkpoint_path):
            try:
                import torch
                from agents.ppo_agent import PokemonActorCritic
                self.device = torch.device("cpu")
                self.model  = PokemonActorCritic().to(self.device)
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model"])
                self.model.eval()
                self.hx = self.cx = None
                print(f"[INFO] Loaded PPO checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint: {e}")
                self.model = None

    def act(self, obs: Dict) -> int:
        if self.model is None:
            return _random_action()
        import torch
        screen = torch.tensor(obs["screen"],     dtype=torch.uint8).unsqueeze(0)
        gs     = torch.tensor(obs["game_state"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _, self.hx, self.cx = self.model.act(
                screen, gs, self.hx, self.cx, deterministic=True
            )
        return int(action.item())

    def reset(self):
        self.hx = self.cx = None


# ── Structured logging (MANDATORY FORMAT) ────────────────────────────────
def log_start(run_id: str, tasks: List[str], config: Dict):
    record = {
        "event":   "START",
        "run_id":  run_id,
        "tasks":   tasks,
        "config":  config,
        "time":    time.time(),
    }
    print(f"[START] {json.dumps(record)}", flush=True)


def log_step(run_id: str, task: str, step: int, action: int,
             reward: float, cumulative: float, state: Dict):
    record = {
        "event":      "STEP",
        "run_id":     run_id,
        "task":       task,
        "step":       step,
        "action":     action,
        "reward":     round(reward, 6),
        "cumulative": round(cumulative, 6),
        "state":      {k: v for k, v in state.items() if k != "game_state"},
        "time":       time.time(),
    }
    print(f"[STEP] {json.dumps(record)}", flush=True)


def log_end(run_id: str, scores: Dict[str, float], total_time: float):
    record = {
        "event":      "END",
        "run_id":     run_id,
        "scores":     {t: round(s, 6) for t, s in scores.items()},
        "total_time": round(total_time, 3),
        "time":       time.time(),
    }
    print(f"[END] {json.dumps(record)}", flush=True)


# ── Run one task episode ──────────────────────────────────────────────────
def run_task(
    task:      str,
    env,
    policy:    Policy,
    run_id:    str,
    max_steps: int = MAX_STEPS,
    llm_freq:  int = 50,
) -> List[Dict]:
    """Run one episode; return trajectory for grading."""
    policy.reset()
    obs  = env.reset()
    traj = []
    cumulative = 0.0

    for step in range(max_steps):
        # LLM advisory every N steps
        if step % llm_freq == 0:
            action = llm_action_hint(env.state(), task)
        else:
            action = policy.act(obs)

        obs2, reward, done, info = env.step(action)
        cumulative += reward

        traj.append({
            "obs":    obs2,
            "action": action,
            "reward": reward,
            "done":   done,
            "info":   info,
        })

        if step % 20 == 0 or done:
            log_step(run_id, task, step, action, reward, cumulative, env.state())

        obs = obs2
        if done:
            break

    return traj


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub",       action="store_true", default=USE_STUB)
    parser.add_argument("--max_steps",  type=int, default=MAX_STEPS)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT)
    parser.add_argument("--tasks",      nargs="*", default=list(GRADER_REGISTRY.keys()))
    args = parser.parse_args()

    run_id  = f"pokemon_rl_{int(time.time())}"
    t_start = time.time()

    config = {
        "model_name":   MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "max_steps":    args.max_steps,
        "stub_mode":    args.stub,
        "checkpoint":   args.checkpoint,
        "tasks":        args.tasks,
    }

    log_start(run_id, args.tasks, config)

    # Build env
    if args.stub or not os.path.exists(ROM_PATH):
        print("[INFO] Using stub environment (no ROM required)", flush=True)
        EnvClass = PokemonRedStubEnv
        env_kwargs = {"max_steps": args.max_steps}
    else:
        EnvClass   = PokemonRedEnv
        env_kwargs = {"rom_path": ROM_PATH, "max_steps": args.max_steps}

    policy = Policy(args.checkpoint)
    scores: Dict[str, float] = {}

    for task in args.tasks:
        print(f"\n[INFO] Running task: {task}", flush=True)
        grader = GRADER_REGISTRY.get(task)
        if grader is None:
            print(f"[WARN] Unknown task {task}, skipping.", flush=True)
            continue

        env = EnvClass(task=task, **{k: v for k, v in env_kwargs.items() if k != "task"} if "task" not in env_kwargs else env_kwargs)

        try:
            traj  = run_task(task, env, policy, run_id, args.max_steps)
            score = grader.grade(traj)
        except Exception as e:
            print(f"[ERROR] Task {task} failed: {e}", flush=True)
            traceback.print_exc()
            score = 0.0
        finally:
            env.close()

        # Clamp to [0, 1]
        score = float(max(0.0, min(1.0, score)))
        scores[task] = score
        print(f"[SCORE] {task} = {score:.4f}", flush=True)

    log_end(run_id, scores, time.time() - t_start)

    # Exit code 0 on success
    sys.exit(0)


if __name__ == "__main__":
    main()
