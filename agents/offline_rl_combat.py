"""
agents/offline_rl_combat.py — Offline RL for Pokemon Battle System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The battle state space is far smaller than the overworld, making it
ideal for offline RL: we learn from a fixed dataset of past battles
rather than requiring live environment interaction.

Algorithm: Conservative Q-Learning (CQL) — a simple offline RL method
that penalises Q-values for out-of-distribution actions, preventing
the agent from overestimating unseen battle moves.

Battle state vector (16-d):
  [my_hp, my_maxhp, my_level, my_type1, my_type2,
   opp_hp, opp_maxhp, opp_level, opp_type1, opp_type2,
   move1_power, move2_power, move3_power, move4_power,
   pp_ratio, turn_number]

Actions: 0=move1, 1=move2, 2=move3, 3=move4, 4=item, 5=switch, 6=run
"""

from __future__ import annotations
import os
import json
import random
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BATTLE_STATE_DIM = 16
N_BATTLE_ACTIONS = 7   # 4 moves + item + switch + run

# Pokémon type chart (simplified — 18 types indexed 0-17)
TYPE_CHART = np.ones((18, 18), dtype=np.float32)
# A few key super-effective pairs (attacker_type, defender_type) → 2.0
_SE = [
    (10, 0), (10, 5), (11, 1), (11, 6), (12, 2), (12, 7),
    (0, 13), (1, 14), (2, 15), (3, 8), (4, 9), (5, 10),
]
for at, dt in _SE:
    TYPE_CHART[at, dt] = 2.0
# Not-very-effective
_NVE = [(at, dt) for dt, at in _SE]
for at, dt in _NVE:
    TYPE_CHART[at, dt] = 0.5


# ── Battle state encoder ──────────────────────────────────────────────────
def encode_battle_state(
    my_hp:    int, my_maxhp: int, my_level: int, my_type: int,
    opp_hp:   int, opp_maxhp: int, opp_level: int, opp_type: int,
    move_powers: List[int],
    pp_remaining: List[int],
    turn: int,
) -> np.ndarray:
    vec = np.zeros(BATTLE_STATE_DIM, dtype=np.float32)
    vec[0]  = my_hp      / max(my_maxhp, 1)
    vec[1]  = my_level   / 100.0
    vec[2]  = my_type    / 17.0
    vec[3]  = opp_hp     / max(opp_maxhp, 1)
    vec[4]  = opp_level  / 100.0
    vec[5]  = opp_type   / 17.0
    for i, (pwr, pp) in enumerate(zip(move_powers[:4], pp_remaining[:4])):
        vec[6 + i]  = pwr / 150.0
        vec[10 + i] = pp  / 40.0
    vec[14] = min(turn, 30) / 30.0
    # Type advantage for move slot 0
    vec[15] = TYPE_CHART[my_type % 18, opp_type % 18] - 1.0
    return vec


# ── Synthetic battle dataset generator ───────────────────────────────────
def generate_synthetic_dataset(n_battles: int = 500, seed: int = 42) -> List[Dict]:
    """
    Generate a synthetic dataset of (state, action, reward, next_state, done)
    tuples representing Pokémon battles. Used when no real game data is available.
    The heuristic policy picks the highest-power move with PP remaining.
    """
    rng = random.Random(seed)
    dataset = []

    for _ in range(n_battles):
        # Random battle setup
        my_level   = rng.randint(5, 55)
        opp_level  = rng.randint(max(1, my_level - 10), my_level + 10)
        my_type    = rng.randint(0, 17)
        opp_type   = rng.randint(0, 17)
        my_hp = my_maxhp = rng.randint(20, 100) + my_level * 2
        opp_hp = opp_maxhp = rng.randint(20, 100) + opp_level * 2
        move_powers = [rng.choice([35, 40, 50, 65, 80, 90, 100, 0]) for _ in range(4)]
        pp_rem      = [rng.randint(0, 35) for _ in range(4)]
        turn = 0

        while my_hp > 0 and opp_hp > 0 and turn < 30:
            state = encode_battle_state(
                my_hp, my_maxhp, my_level, my_type,
                opp_hp, opp_maxhp, opp_level, opp_type,
                move_powers, pp_rem, turn,
            )

            # Heuristic: pick best available move
            best_action = 0
            best_score  = -1.0
            for i in range(4):
                if pp_rem[i] > 0 and move_powers[i] > 0:
                    type_adv = TYPE_CHART[my_type % 18, opp_type % 18]
                    score    = move_powers[i] * type_adv
                    if score > best_score:
                        best_score  = score
                        best_action = i

            # Add some noise
            if rng.random() < 0.1:
                best_action = rng.randint(0, 3)

            # Simulate outcome
            damage_to_opp = int(
                (my_level / 5 * move_powers[best_action] * (my_level / opp_level))
                * TYPE_CHART[my_type % 18, opp_type % 18]
                * rng.uniform(0.85, 1.0) / 50
            )
            damage_to_me = int(
                opp_level * rng.randint(20, 60) * rng.uniform(0.85, 1.0) / 100
            ) if opp_hp > 0 else 0

            opp_hp = max(0, opp_hp - damage_to_opp)
            my_hp  = max(0, my_hp  - damage_to_me)
            if pp_rem[best_action] > 0:
                pp_rem[best_action] -= 1

            reward = 0.0
            if opp_hp == 0:
                reward = 1.0   # knocked out opponent
            elif my_hp == 0:
                reward = -0.5  # fainted
            else:
                reward = damage_to_opp / max(opp_maxhp, 1) * 0.1

            done = (my_hp == 0 or opp_hp == 0)

            next_state = encode_battle_state(
                my_hp, my_maxhp, my_level, my_type,
                opp_hp, opp_maxhp, opp_level, opp_type,
                move_powers, pp_rem, turn + 1,
            )

            dataset.append({
                "state":      state.tolist(),
                "action":     best_action,
                "reward":     reward,
                "next_state": next_state.tolist(),
                "done":       done,
            })

            turn += 1
            if done:
                break

    return dataset


# ── CQL Q-Network ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class BattleQNetwork(nn.Module):
        def __init__(self, state_dim: int = BATTLE_STATE_DIM, n_actions: int = N_BATTLE_ACTIONS):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


class CQLBattleTrainer:
    """
    Conservative Q-Learning for offline battle strategy.
    Penalises Q-values on actions not seen in the dataset
    to avoid over-optimistic extrapolation.
    """

    def __init__(
        self,
        lr:          float = 1e-3,
        gamma:       float = 0.99,
        cql_alpha:   float = 1.0,
        batch_size:  int   = 64,
        n_epochs:    int   = 100,
        checkpoint_dir: str = "checkpoints/combat",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not installed")

        import torch
        self.gamma     = gamma
        self.cql_alpha = cql_alpha
        self.bs        = batch_size
        self.epochs    = n_epochs
        self.ckpt_dir  = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device    = torch.device("cpu")

        self.q_net     = BattleQNetwork().to(self.device)
        self.q_target  = BattleQNetwork().to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.opt       = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.training_log: List[Dict] = []

    def _batch_to_tensors(self, batch: List[Dict]):
        import torch
        states      = torch.tensor([b["state"]      for b in batch], dtype=torch.float32)
        actions     = torch.tensor([b["action"]     for b in batch], dtype=torch.long)
        rewards     = torch.tensor([b["reward"]     for b in batch], dtype=torch.float32)
        next_states = torch.tensor([b["next_state"] for b in batch], dtype=torch.float32)
        dones       = torch.tensor([b["done"]       for b in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def train(self, dataset: List[Dict]) -> List[Dict]:
        import torch
        print(f"\n[CQL] Training on {len(dataset)} transitions for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            random.shuffle(dataset)
            total_loss = td_loss = cql_loss = 0.0
            n_batches  = 0

            for start in range(0, len(dataset), self.bs):
                batch = dataset[start:start + self.bs]
                if len(batch) < 4:
                    continue

                s, a, r, s2, d = self._batch_to_tensors(batch)

                # TD target
                with torch.no_grad():
                    next_q  = self.q_target(s2)
                    max_q2  = next_q.max(dim=1).values
                    target  = r + self.gamma * (1 - d) * max_q2

                # Current Q values
                q_vals      = self.q_net(s)
                q_taken     = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

                td  = F.mse_loss(q_taken, target)

                # CQL penalty: push down Q on ALL actions, pull up on taken
                log_sum_exp = torch.logsumexp(q_vals, dim=1).mean()
                cql         = self.cql_alpha * (log_sum_exp - q_taken.mean())

                loss = td + cql
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += loss.item()
                td_loss    += td.item()
                cql_loss   += cql.item()
                n_batches  += 1

            # Soft update target
            for tp, p in zip(self.q_target.parameters(), self.q_net.parameters()):
                tp.data.copy_(0.995 * tp.data + 0.005 * p.data)

            nb = max(1, n_batches)
            log = {
                "epoch":    epoch,
                "loss":     total_loss / nb,
                "td_loss":  td_loss    / nb,
                "cql_loss": cql_loss   / nb,
            }
            self.training_log.append(log)

            if epoch % 20 == 0:
                print(f"  [CQL epoch {epoch:4d}] loss={log['loss']:.4f} "
                      f"td={log['td_loss']:.4f} cql={log['cql_loss']:.4f}")

        # Save
        import torch
        ckpt = self.ckpt_dir / "battle_cql.pt"
        torch.save(self.q_net.state_dict(), ckpt)
        print(f"[CQL] Saved combat Q-network → {ckpt}")
        return self.training_log

    def act(self, state_vec: np.ndarray, epsilon: float = 0.0) -> int:
        """Select best action from Q-network (with optional epsilon-greedy)."""
        import torch
        if random.random() < epsilon:
            return random.randint(0, N_BATTLE_ACTIONS - 1)
        s = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(s)
        return int(q.argmax(dim=1).item())

    def load(self, path: str):
        import torch
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.q_net.eval()


# ── Dataset persistence ───────────────────────────────────────────────────
def save_dataset(dataset: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataset, f)
    print(f"[CQL] Dataset saved: {len(dataset)} transitions → {path}")


def load_dataset(path: str) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    print(f"[CQL] Dataset loaded: {len(data)} transitions ← {path}")
    return data


# ── Standalone train ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_battles", type=int, default=500)
    parser.add_argument("--epochs",    type=int, default=100)
    parser.add_argument("--dataset",   type=str, default="")
    args = parser.parse_args()

    if args.dataset and os.path.exists(args.dataset):
        dataset = load_dataset(args.dataset)
    else:
        print(f"[CQL] Generating {args.n_battles} synthetic battles...")
        dataset = generate_synthetic_dataset(args.n_battles)
        save_dataset(dataset, "data/synthetic_battles.json")

    trainer = CQLBattleTrainer(n_epochs=args.epochs)
    trainer.train(dataset)
