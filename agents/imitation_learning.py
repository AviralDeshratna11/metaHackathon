"""
agents/imitation_learning.py — Behavioural Cloning + DAgger
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Addresses the "short-sighted" behaviour problem: agents that
grind XP instead of progressing through gyms.

Two modes:
1. Behavioural Cloning (BC) — pure supervised learning on demonstrations.
   Fast but suffers from distribution shift.

2. DAgger (Dataset Aggregation) — iteratively mixes agent rollouts
   with expert labels, fixing distribution shift over time.

Expert data can come from:
  a) Recorded human playthroughs (input logs → action sequences)
  b) Scripted "oracle" policy that follows the optimal gym path
  c) LLM-labelled states (using OpenAI client per hackathon spec)
"""

from __future__ import annotations
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Oracle / scripted expert ──────────────────────────────────────────────
class OraclePolicy:
    """
    Rule-based expert that encodes high-level Pokémon Red game logic.
    Used to generate demonstration data when human recordings are absent.
    
    Strategy:
      - If HP < 30%: use a healing item (action=4) or run (action=6)
      - If in battle position: prefer highest-power move
      - If exploring: move toward next waypoint (Pallet→Viridian→Pewter→...)
    """

    WAYPOINT_SEQUENCE = [
        # (map_id_approx, preferred_direction)
        (1,  3),  # Pallet Town → go up
        (2,  3),  # Route 1 → go up
        (4,  4),  # Viridian City → press A to advance
        (5,  3),  # Route 2 → go up
        (6,  4),  # Viridian Forest → press A through fights
        (7,  3),  # Route 3 (Pewter outskirts) → go up
        (8,  4),  # Pewter City (go to gym) → press A
        (9,  2),  # Route 4 → go right
        (11, 2),  # Mt. Moon → go right (persist!)
        (12, 2),  # Mt. Moon deep → go right
        (3,  2),  # Cerulean City → arrived!
    ]

    def act(self, obs: Dict) -> int:
        gs  = obs.get("game_state", np.zeros(32, dtype=np.float32))
        hp_ratio = gs[2] if len(gs) > 2 else 1.0
        map_id   = round(gs[3] * 255) if len(gs) > 3 else 0

        # Low HP → try to use item or run
        if hp_ratio < 0.25:
            return 4  # press A (often "use item" in menu context)

        # Find preferred direction for current map
        for wmap, wdir in self.WAYPOINT_SEQUENCE:
            if abs(map_id - wmap) <= 1:
                return wdir

        # Default: keep moving right / up alternating with A presses
        return random.choice([2, 3, 4])


# ── Demonstration dataset ─────────────────────────────────────────────────
class DemoDataset:
    """Stores (screen, game_state, action) demonstration tuples."""

    def __init__(self, max_size: int = 50_000):
        self.max_size   = max_size
        self.screens:    List[np.ndarray] = []
        self.gstates:    List[np.ndarray] = []
        self.actions:    List[int]        = []

    def add(self, screen: np.ndarray, gs: np.ndarray, action: int):
        if len(self.actions) >= self.max_size:
            idx = random.randrange(self.max_size)
            self.screens[idx]  = screen
            self.gstates[idx]  = gs
            self.actions[idx]  = action
        else:
            self.screens.append(screen)
            self.gstates.append(gs)
            self.actions.append(action)

    def add_trajectory(self, trajectory: List[Dict], expert_actions: List[int]):
        for step, action in zip(trajectory, expert_actions):
            obs = step.get("obs", {})
            self.add(obs["screen"], obs["game_state"], action)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx     = random.sample(range(len(self.actions)), min(batch_size, len(self.actions)))
        screens = np.array([self.screens[i] for i in idx], dtype=np.uint8)
        gstates = np.array([self.gstates[i] for i in idx], dtype=np.float32)
        actions = np.array([self.actions[i] for i in idx], dtype=np.int64)
        return screens, gstates, actions

    def __len__(self):
        return len(self.actions)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            screens=np.array(self.screens,  dtype=np.uint8),
            gstates=np.array(self.gstates,  dtype=np.float32),
            actions=np.array(self.actions,  dtype=np.int64),
        )
        print(f"[IL] Demo dataset saved: {len(self)} steps → {path}")

    @classmethod
    def load(cls, path: str) -> "DemoDataset":
        data    = np.load(path)
        dataset = cls()
        dataset.screens = list(data["screens"])
        dataset.gstates = list(data["gstates"])
        dataset.actions = list(data["actions"])
        print(f"[IL] Demo dataset loaded: {len(dataset)} steps ← {path}")
        return dataset


# ── LLM expert labeller (uses OpenAI client per spec) ────────────────────
class LLMExpertLabeller:
    """
    Uses the hackathon-mandated OpenAI client to label states with
    expert actions. Falls back to oracle policy on failure.
    """

    SYSTEM_PROMPT = """You are an expert Pokémon Red speedrunner.
Given a game state, choose the single best action to progress toward the next gym badge.
Respond ONLY with the integer action number (0-7):
  0=down, 1=left, 2=right, 3=up, 4=A, 5=B, 6=start, 7=select
Consider: exploration progress, HP ratio, map location, badge count.
Prefer movement toward the next gym over grinding."""

    def __init__(self, api_base_url: str = "", model: str = "", token: str = ""):
        self.api_base_url = api_base_url or os.environ.get("API_BASE_URL", "")
        self.model        = model        or os.environ.get("MODEL_NAME", "gpt-4o-mini")
        self.token        = token        or os.environ.get("HF_TOKEN", "")
        self.oracle       = OraclePolicy()
        self._client      = None

        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.token or "dummy",
                base_url=self.api_base_url or "https://api.openai.com/v1",
            )
        except Exception:
            pass

    def label(self, obs: Dict, env_state: Dict) -> int:
        if self._client is None:
            return self.oracle.act(obs)

        gs = obs.get("game_state", [])
        try:
            user_msg = json.dumps({
                "hp_ratio":    round(float(gs[2]), 3) if len(gs) > 2 else 1.0,
                "map_id":      round(float(gs[3]) * 255) if len(gs) > 3 else 0,
                "x":           round(float(gs[4]) * 255) if len(gs) > 4 else 0,
                "y":           round(float(gs[5]) * 255) if len(gs) > 5 else 0,
                "level":       round(float(gs[1]) * 100) if len(gs) > 1 else 5,
                "badges":      env_state.get("badges", 0),
                "tiles_seen":  env_state.get("tiles_visited", 0),
            })
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            action = int(resp.choices[0].message.content.strip())
            if 0 <= action <= 7:
                return action
        except Exception:
            pass
        return self.oracle.act(obs)


# ── Behavioural Cloning trainer ───────────────────────────────────────────
if TORCH_AVAILABLE:
    class BCTrainer:
        """
        Supervised learning on demonstration data.
        Trains the PPO actor-critic's policy head to mimic the expert.
        """

        def __init__(
            self,
            lr:         float = 3e-4,
            batch_size: int   = 64,
            n_epochs:   int   = 20,
            checkpoint_dir: str = "checkpoints/il",
        ):
            import torch
            from agents.ppo_agent import PokemonActorCritic
            self.bs       = batch_size
            self.epochs   = n_epochs
            self.ckpt_dir = Path(checkpoint_dir)
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.device   = torch.device("cpu")
            self.model    = PokemonActorCritic().to(self.device)
            self.opt      = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.log: List[Dict] = []

        def train(self, dataset: DemoDataset) -> List[Dict]:
            import torch
            print(f"\n[BC] Training on {len(dataset)} demos for {self.epochs} epochs...")
            N = len(dataset)

            for epoch in range(self.epochs):
                indices   = list(range(N))
                random.shuffle(indices)
                total_loss = acc = n_batches = 0

                for start in range(0, N, self.bs):
                    batch_idx = indices[start:start + self.bs]
                    screens   = np.array([dataset.screens[i] for i in batch_idx], dtype=np.uint8)
                    gstates   = np.array([dataset.gstates[i] for i in batch_idx], dtype=np.float32)
                    actions   = np.array([dataset.actions[i] for i in batch_idx], dtype=np.int64)

                    s_sc  = torch.tensor(screens, dtype=torch.uint8).to(self.device)
                    s_gs  = torch.tensor(gstates, dtype=torch.float32).to(self.device)
                    s_act = torch.tensor(actions, dtype=torch.long).to(self.device)

                    logits, _, _, _ = self.model(s_sc, s_gs)
                    loss = F.cross_entropy(logits, s_act)

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()

                    pred       = logits.argmax(dim=1)
                    acc       += (pred == s_act).float().mean().item()
                    total_loss += loss.item()
                    n_batches  += 1

                nb  = max(1, n_batches)
                log = {
                    "epoch":    epoch,
                    "loss":     total_loss / nb,
                    "accuracy": acc / nb,
                }
                self.log.append(log)

                if epoch % 5 == 0:
                    print(f"  [BC epoch {epoch:3d}] loss={log['loss']:.4f} acc={log['accuracy']:.3f}")

            ckpt = self.ckpt_dir / "bc_policy.pt"
            torch.save(self.model.state_dict(), ckpt)
            print(f"[BC] Policy saved → {ckpt}")
            return self.log

        def get_model(self):
            return self.model


# ── DAgger ────────────────────────────────────────────────────────────────
class DAggerTrainer:
    """
    Dataset Aggregation (DAgger):
    1. Train BC on initial demos.
    2. Roll out current policy.
    3. Label roll-out states with expert actions.
    4. Add to dataset, re-train.
    5. Repeat.

    After N rounds, the policy learns to handle its own state distribution.
    """

    def __init__(
        self,
        env_factory,
        expert: "OraclePolicy | LLMExpertLabeller",
        n_rounds:     int   = 5,
        steps_per_round: int = 200,
        beta_decay:   float = 0.7,   # mix of expert vs learned policy
        checkpoint_dir: str = "checkpoints/dagger",
    ):
        self.env_factory     = env_factory
        self.expert          = expert
        self.n_rounds        = n_rounds
        self.steps_per_round = steps_per_round
        self.beta            = 1.0   # start fully expert
        self.beta_decay      = beta_decay
        self.ckpt_dir        = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.dataset         = DemoDataset()
        self.bc_trainer      = BCTrainer(checkpoint_dir=str(self.ckpt_dir)) if TORCH_AVAILABLE else None
        self.round_logs: List[Dict] = []

    def _collect_round(self, policy=None) -> int:
        """Collect steps with beta-mix of expert and learned policy."""
        env  = self.env_factory()
        obs  = env.reset()
        added = 0

        for step in range(self.steps_per_round):
            # Beta: probability of using expert
            use_expert = (random.random() < self.beta) or (policy is None)

            if use_expert:
                action = self.expert.act(obs)
            else:
                try:
                    import torch
                    screen = torch.tensor(obs["screen"],     dtype=torch.uint8).unsqueeze(0)
                    gs     = torch.tensor(obs["game_state"], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action_t, *_ = policy.act(screen, gs)
                    action = int(action_t.item())
                except Exception:
                    action = self.expert.act(obs)

            # Always label with expert (DAgger key idea)
            expert_action = self.expert.act(obs)
            self.dataset.add(obs["screen"], obs["game_state"], expert_action)
            added += 1

            obs2, _, done, info = env.step(action)
            obs = obs2
            if done:
                obs = env.reset()

        env.close()
        return added

    def train(self) -> List[Dict]:
        print(f"\n[DAgger] {self.n_rounds} rounds × {self.steps_per_round} steps each")
        policy = None

        for round_i in range(self.n_rounds):
            print(f"\n[DAgger round {round_i}] beta={self.beta:.3f} dataset={len(self.dataset)}")

            added = self._collect_round(policy)

            if self.bc_trainer and len(self.dataset) >= 32:
                bc_log = self.bc_trainer.train(self.dataset)
                final_loss = bc_log[-1]["loss"] if bc_log else 0.0
                final_acc  = bc_log[-1]["accuracy"] if bc_log else 0.0
                policy     = self.bc_trainer.get_model()
            else:
                final_loss = final_acc = 0.0

            log = {
                "round":        round_i,
                "beta":         self.beta,
                "steps_added":  added,
                "dataset_size": len(self.dataset),
                "bc_loss":      final_loss,
                "bc_accuracy":  final_acc,
            }
            self.round_logs.append(log)
            print(f"  → loss={final_loss:.4f}  acc={final_acc:.3f}  dataset={len(self.dataset)}")

            # Decay beta: rely less on expert over time
            self.beta *= self.beta_decay

        # Save final dataset
        self.dataset.save(str(self.ckpt_dir / "dagger_dataset.npz"))
        return self.round_logs


# ── Convenience runner ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from envs.pokemon_red_env import PokemonRedStubEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["bc", "dagger"], default="dagger")
    parser.add_argument("--rounds",  type=int, default=3)
    parser.add_argument("--steps",   type=int, default=100)
    args = parser.parse_args()

    expert  = OraclePolicy()
    env_fac = lambda: PokemonRedStubEnv(max_steps=args.steps)

    if args.mode == "dagger":
        trainer = DAggerTrainer(
            env_factory=env_fac,
            expert=expert,
            n_rounds=args.rounds,
            steps_per_round=args.steps,
        )
        logs = trainer.train()
    else:
        dataset = DemoDataset()
        env     = env_fac()
        obs     = env.reset()
        for _ in range(args.steps):
            a   = expert.act(obs)
            dataset.add(obs["screen"], obs["game_state"], a)
            obs2, _, done, _ = env.step(a)
            obs = env.reset() if done else obs2
        env.close()
        if TORCH_AVAILABLE:
            trainer = BCTrainer()
            logs    = trainer.train(dataset)
        else:
            logs = []

    print(json.dumps(logs, indent=2))
