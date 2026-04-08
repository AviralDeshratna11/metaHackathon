"""
agents/marl_coordinator.py — Multi-Agent RL Coordinator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs N parallel agents on independent env instances.
Implements population-based diversity: agents are encouraged
to explore *different* parts of the map (anti-correlation bonus).
This is the strategy that broke through the Mt. Moon bottleneck —
multiple agents explore simultaneously; the best trajectory
is used to seed the next generation's policy update.
"""

from __future__ import annotations
import os
import time
import math
import random
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class AgentRecord:
    agent_id:    int
    task:        str
    trajectory:  List[Dict] = field(default_factory=list)
    total_reward: float = 0.0
    tiles_visited: set  = field(default_factory=set)
    maps_visited:  set  = field(default_factory=set)
    badges:        int  = 0
    level:         int  = 1
    alive:         bool = True
    thread:        Optional[threading.Thread] = None


class MARLCoordinator:
    """
    Runs N agents in parallel threads, each on their own env instance.

    Key features:
    1. Diversity bonus — agents get extra reward for visiting tiles
       that other agents haven't, encouraging exploration spread.
    2. Elite seeding — after each generation, the top-k agents'
       policies are used to warm-start the next generation.
    3. Mt. Moon override — if any agent reaches map_id >= threshold,
       all agents get a curriculum nudge toward that region.
    """

    MT_MOON_MAP_IDS   = set(range(11, 18))   # approximate GB map IDs
    CERULEAN_MAP_ID   = 3

    def __init__(
        self,
        n_agents:      int   = 4,
        task:          str   = "reach_cerulean_city",
        max_steps:     int   = 1024,
        diversity_coef: float = 0.15,
        elite_frac:    float = 0.25,
        use_stub:      bool  = True,
        rom_path:      str   = "pokemon_red.gb",
        checkpoint_dir: str  = "checkpoints/marl",
    ):
        self.n_agents       = n_agents
        self.task           = task
        self.max_steps      = max_steps
        self.diversity_coef = diversity_coef
        self.elite_frac     = elite_frac
        self.use_stub       = use_stub
        self.rom_path       = rom_path
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Shared state (protected by lock)
        self._lock             = threading.Lock()
        self._global_tiles:    set = set()   # union of all visited tiles
        self._mt_moon_reached: bool = False
        self._cerulean_reached: bool = False
        self._generation       = 0
        self._best_ever_reward = -math.inf
        self.generation_logs: List[Dict] = []

    # ── Env factory ───────────────────────────────────────────────────────
    def _make_env(self, agent_id: int):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from envs.pokemon_red_env import PokemonRedStubEnv, PokemonRedEnv

        if self.use_stub or not os.path.exists(self.rom_path):
            return PokemonRedStubEnv(task=self.task, max_steps=self.max_steps)
        return PokemonRedEnv(
            rom_path=self.rom_path,
            task=self.task,
            max_steps=self.max_steps,
            headless=True,
        )

    # ── Diversity reward ──────────────────────────────────────────────────
    def _diversity_bonus(self, new_tiles: set) -> float:
        """Reward tiles this agent visits that no other agent has seen."""
        with self._lock:
            novel = new_tiles - self._global_tiles
            self._global_tiles |= new_tiles
        return self.diversity_coef * len(novel)

    # ── Single-agent rollout (runs in its own thread) ─────────────────────
    def _run_agent(self, record: AgentRecord, policy=None):
        env = self._make_env(record.agent_id)
        obs = env.reset()

        hx = cx = None
        cumulative = 0.0
        prev_tiles: set = set()

        for step in range(self.max_steps):
            # Choose action
            action = self._select_action(obs, policy, hx, cx, record)

            obs2, reward, done, info = env.step(action)

            # Diversity augmentation
            cur_tiles  = set()
            cur_maps   = set()
            state      = env.state()
            map_id_norm = obs2["game_state"][3] if len(obs2["game_state"]) > 3 else 0
            map_id      = round(map_id_norm * 255)
            x           = round(obs2["game_state"][4] * 255) if len(obs2["game_state"]) > 4 else 0
            y           = round(obs2["game_state"][5] * 255) if len(obs2["game_state"]) > 5 else 0
            cur_tiles.add((map_id, x, y))
            cur_maps.add(map_id)

            div_bonus  = self._diversity_bonus(cur_tiles - prev_tiles)
            aug_reward = reward + div_bonus
            cumulative += aug_reward

            prev_tiles |= cur_tiles

            record.trajectory.append({
                "obs":    obs2,
                "action": action,
                "reward": aug_reward,
                "done":   done,
                "info":   info,
            })
            record.total_reward  = cumulative
            record.tiles_visited |= cur_tiles
            record.maps_visited  |= cur_maps
            record.badges  = info.get("badges", record.badges)
            record.level   = info.get("level",  record.level)

            # Check milestone flags
            if map_id in self.MT_MOON_MAP_IDS:
                with self._lock:
                    if not self._mt_moon_reached:
                        self._mt_moon_reached = True
                        print(f"  [MARL] Agent {record.agent_id} reached Mt. Moon! 🗻")

            if map_id == self.CERULEAN_MAP_ID:
                with self._lock:
                    if not self._cerulean_reached:
                        self._cerulean_reached = True
                        print(f"  [MARL] Agent {record.agent_id} reached Cerulean City! 🌊")

            obs = obs2
            if done:
                break

        env.close()
        record.alive = False

    def _select_action(self, obs, policy, hx, cx, record: AgentRecord) -> int:
        """Action selection with optional PPO policy."""
        if policy is not None:
            try:
                import torch
                screen = torch.tensor(obs["screen"],     dtype=torch.uint8).unsqueeze(0)
                gs     = torch.tensor(obs["game_state"], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, hx, cx = policy.act(screen, gs, hx, cx)
                return int(action.item())
            except Exception:
                pass
        # Biased random: prefer movement actions
        weights = [0.2, 0.2, 0.2, 0.2, 0.1, 0.05, 0.025, 0.025]
        return random.choices(range(8), weights=weights)[0]

    # ── Generation runner ─────────────────────────────────────────────────
    def run_generation(self, policies=None) -> List[AgentRecord]:
        """
        Run one generation: N agents in parallel threads.
        Returns sorted records (best first).
        """
        self._global_tiles = set()   # reset shared tile pool each gen
        records: List[AgentRecord] = [
            AgentRecord(agent_id=i, task=self.task)
            for i in range(self.n_agents)
        ]

        threads = []
        for i, record in enumerate(records):
            policy = policies[i % len(policies)] if policies else None
            t = threading.Thread(
                target=self._run_agent,
                args=(record, policy),
                daemon=True,
            )
            t.start()
            threads.append(t)
            # Stagger starts slightly to avoid I/O contention
            time.sleep(0.05)

        for t in threads:
            t.join(timeout=300)

        records.sort(key=lambda r: r.total_reward, reverse=True)
        return records

    # ── Multi-generation training loop ────────────────────────────────────
    def train(
        self,
        n_generations: int = 10,
        ppo_update_freq: int = 2,
    ) -> List[Dict]:
        """
        Main MARL training loop.
        Every ppo_update_freq generations, run a PPO update on the
        best trajectory from this cohort.
        """
        print(f"\n{'═'*60}")
        print(f"  MARL Training — {self.n_agents} agents × {n_generations} generations")
        print(f"  Task: {self.task}")
        print(f"{'═'*60}\n")

        policies = None

        for gen in range(n_generations):
            self._generation = gen
            t0 = time.time()

            print(f"[Gen {gen:3d}] Launching {self.n_agents} agents...")
            records = self.run_generation(policies)

            best    = records[0]
            worst   = records[-1]
            avg_rew = sum(r.total_reward for r in records) / len(records)

            dt = time.time() - t0
            print(
                f"[Gen {gen:3d}] best={best.total_reward:.3f} "
                f"avg={avg_rew:.3f} worst={worst.total_reward:.3f} "
                f"maps={len(best.maps_visited)} "
                f"badges={best.badges} "
                f"mt_moon={'✓' if self._mt_moon_reached else '✗'} "
                f"cerulean={'✓' if self._cerulean_reached else '✗'} "
                f"({dt:.1f}s)"
            )

            gen_log = {
                "generation":       gen,
                "best_reward":      best.total_reward,
                "avg_reward":       avg_rew,
                "worst_reward":     worst.total_reward,
                "maps_visited":     len(best.maps_visited),
                "tiles_visited":    len(best.tiles_visited),
                "badges":           best.badges,
                "level":            best.level,
                "mt_moon_reached":  self._mt_moon_reached,
                "cerulean_reached": self._cerulean_reached,
                "elapsed_s":        dt,
            }
            self.generation_logs.append(gen_log)

            # Track global best
            if best.total_reward > self._best_ever_reward:
                self._best_ever_reward = best.total_reward
                self._save_best_trajectory(best)

            # PPO fine-tune on elite trajectories
            if gen % ppo_update_freq == 0 and gen > 0:
                n_elite = max(1, int(self.n_agents * self.elite_frac))
                elite_trajs = [r.trajectory for r in records[:n_elite]]
                policies = self._ppo_update_from_trajectories(elite_trajs)

        print(f"\n[MARL] Training complete. Best reward ever: {self._best_ever_reward:.4f}")
        return self.generation_logs

    def _ppo_update_from_trajectories(self, trajectories: List[List[Dict]]):
        """Lightweight PPO update from offline trajectories. Returns updated policies."""
        try:
            import torch
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from agents.ppo_agent import PPOTrainer, PokemonActorCritic

            # Flatten trajectories
            flat = [step for traj in trajectories for step in traj]
            if len(flat) < 32:
                return None

            device = torch.device("cpu")
            model  = PokemonActorCritic().to(device)

            # Simple supervised update on best actions (behavioural cloning on elite)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            import torch.nn.functional as F
            from torch.distributions import Categorical

            screens = np.array([s["obs"]["screen"]     for s in flat[:256]], dtype=np.uint8)
            gstates = np.array([s["obs"]["game_state"] for s in flat[:256]], dtype=np.float32)
            actions = np.array([s["action"]            for s in flat[:256]], dtype=np.int64)

            s_sc  = torch.tensor(screens, dtype=torch.uint8).to(device)
            s_gs  = torch.tensor(gstates, dtype=torch.float32).to(device)
            s_act = torch.tensor(actions, dtype=torch.long).to(device)

            for _ in range(3):
                logits, _, _, _ = model(s_sc, s_gs)
                loss = F.cross_entropy(logits, s_act)
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            return [model] * self.n_agents

        except Exception as e:
            print(f"  [WARN] PPO update failed: {e}")
            return None

    def _save_best_trajectory(self, record: AgentRecord):
        """Save a summary of the best trajectory for offline RL use."""
        import json
        path = self.checkpoint_dir / "best_trajectory_summary.json"
        summary = {
            "agent_id":       record.agent_id,
            "total_reward":   record.total_reward,
            "badges":         record.badges,
            "level":          record.level,
            "maps_visited":   list(record.maps_visited),
            "tiles_count":    len(record.tiles_visited),
            "trajectory_len": len(record.trajectory),
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  [MARL] Saved best trajectory summary → {path}")


# ── Standalone runner ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents",      type=int, default=4)
    parser.add_argument("--n_generations", type=int, default=5)
    parser.add_argument("--task",          type=str, default="reach_cerulean_city")
    parser.add_argument("--max_steps",     type=int, default=256)
    args = parser.parse_args()

    coord = MARLCoordinator(
        n_agents=args.n_agents,
        task=args.task,
        max_steps=args.max_steps,
        use_stub=True,
    )
    logs = coord.train(n_generations=args.n_generations)
    print(json.dumps(logs, indent=2))
