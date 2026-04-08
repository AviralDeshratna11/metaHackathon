#!/usr/bin/env python3
"""
train.py — PPO training entry point for Pokemon Red RL agent
Usage:
    python train.py --stub               # fast dev loop (no ROM)
    python train.py --rom pokemon_red.gb # real game
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Train Pokemon Red PPO Agent")
    parser.add_argument("--stub",        action="store_true", default=True,
                        help="Use stub env (no ROM required)")
    parser.add_argument("--rom",         type=str, default="pokemon_red.gb")
    parser.add_argument("--task",        type=str, default="obtain_first_badge")
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--n_steps",     type=int, default=512)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best.pt")
    parser.add_argument("--resume",      type=str, default="")
    parser.add_argument("--log_dir",     type=str, default="logs")
    args = parser.parse_args()

    print("=" * 60)
    print("  Pokémon Red RL — PPO Training")
    print("=" * 60)
    print(f"  Task:        {args.task}")
    print(f"  Total steps: {args.total_steps:,}")
    print(f"  Stub mode:   {args.stub}")
    print()

    # ── Import env ──────────────────────────────────────────────────────
    from envs.pokemon_red_env import PokemonRedStubEnv, PokemonRedEnv

    if args.stub or not os.path.exists(args.rom):
        print("[INFO] Using stub environment")
        env = PokemonRedStubEnv(task=args.task, max_steps=2048)
    else:
        print(f"[INFO] Using real PyBoy environment: {args.rom}")
        env = PokemonRedEnv(
            rom_path=args.rom,
            task=args.task,
            max_steps=2048,
            headless=True,
        )

    # ── Import agent ────────────────────────────────────────────────────
    try:
        from agents.ppo_agent import PPOTrainer
    except ImportError as e:
        print(f"[ERROR] Cannot import PPO agent: {e}")
        print("        Install: pip install torch")
        sys.exit(1)

    trainer = PPOTrainer(
        env=env,
        lr=args.lr,
        n_steps=args.n_steps,
        checkpoint_dir=str(Path(args.checkpoint).parent),
    )

    if args.resume and os.path.exists(args.resume):
        trainer.load_checkpoint(args.resume)

    # ── Train ───────────────────────────────────────────────────────────
    t_start = time.time()
    log = trainer.train(total_steps=args.total_steps)

    elapsed = time.time() - t_start
    print(f"\n[DONE] Training complete in {elapsed/60:.1f} min")
    print(f"       Best reward: {trainer.best_reward:.4f}")
    print(f"       Checkpoint:  {args.checkpoint}")

    # ── Save log ────────────────────────────────────────────────────────
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / f"training_{int(t_start)}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"       Log:         {log_path}")

    env.close()


if __name__ == "__main__":
    main()
