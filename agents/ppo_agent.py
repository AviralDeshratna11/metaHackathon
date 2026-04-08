"""
PPO Agent for Pokemon Red — with CNN encoder + LSTM memory.
Compatible with both real PyBoy env and the stub env.
"""

from __future__ import annotations
import os
import json
import time
import math
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Neural network components ─────────────────────────────────────────────
if TORCH_AVAILABLE:
    class CNNEncoder(nn.Module):
        """Encodes 144×160×3 screen frames → 256-d latent."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4),   # → 35×38
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),  # → 16×18
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),  # → 14×16
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 14 * 16, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, H, W, C) uint8 → (B, C, H, W) float
            x = x.float() / 255.0
            x = x.permute(0, 3, 1, 2)
            return self.net(x)

    class PokemonActorCritic(nn.Module):
        """
        Actor-Critic with:
        - CNN screen encoder
        - linear game-state encoder
        - LSTM memory (128 hidden)
        - policy head + value head
        """
        N_ACTIONS   = 8
        STATE_DIM   = 32
        LSTM_HIDDEN = 128

        def __init__(self):
            super().__init__()
            self.cnn  = CNNEncoder()               # → 256
            self.gs   = nn.Sequential(             # → 64
                nn.Linear(self.STATE_DIM, 64),
                nn.ReLU(),
            )
            self.lstm = nn.LSTMCell(256 + 64, self.LSTM_HIDDEN)
            self.actor  = nn.Linear(self.LSTM_HIDDEN, self.N_ACTIONS)
            self.critic = nn.Linear(self.LSTM_HIDDEN, 1)

        def forward(
            self,
            screen:     "torch.Tensor",   # (B, 144, 160, 3)
            game_state: "torch.Tensor",   # (B, 32)
            hx: Optional["torch.Tensor"] = None,
            cx: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            B = screen.shape[0]
            if hx is None:
                hx = torch.zeros(B, self.LSTM_HIDDEN, device=screen.device)
            if cx is None:
                cx = torch.zeros(B, self.LSTM_HIDDEN, device=screen.device)

            z_cnn = self.cnn(screen)              # (B, 256)
            z_gs  = self.gs(game_state)           # (B, 64)
            z     = torch.cat([z_cnn, z_gs], -1)  # (B, 320)

            hx, cx = self.lstm(z, (hx, cx))
            logits = self.actor(hx)               # (B, 8)
            value  = self.critic(hx).squeeze(-1)  # (B,)
            return logits, value, hx, cx

        def act(
            self,
            screen:     "torch.Tensor",
            game_state: "torch.Tensor",
            hx=None, cx=None, deterministic=False,
        ):
            logits, value, hx, cx = self(screen, game_state, hx, cx)
            dist   = Categorical(logits=logits)
            action = dist.mode if deterministic else dist.sample()
            log_p  = dist.log_prob(action)
            return action, log_p, value, hx, cx


# ── Rollout buffer ────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int = 1):
        self.n_steps = n_steps
        self.n_envs  = n_envs
        self.reset()

    def reset(self):
        self.screens     = []
        self.game_states = []
        self.actions     = []
        self.log_probs   = []
        self.rewards     = []
        self.values      = []
        self.dones       = []

    def add(self, screen, gs, action, log_p, reward, value, done):
        self.screens.append(screen)
        self.game_states.append(gs)
        self.actions.append(action)
        self.log_probs.append(log_p)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[List[float], List[float]]:
        """GAE advantage estimation."""
        returns    = []
        advantages = []
        gae        = 0.0
        next_val   = last_value
        for i in reversed(range(len(self.rewards))):
            mask    = 1.0 - float(self.dones[i])
            delta   = self.rewards[i] + gamma * next_val * mask - self.values[i]
            gae     = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
            next_val = self.values[i]
        return returns, advantages


# ── PPO Trainer ───────────────────────────────────────────────────────────
class PPOTrainer:
    """
    Proximal Policy Optimisation trainer.
    Works with any env that implements reset() / step() / state().
    """

    def __init__(
        self,
        env,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        clip_eps:      float = 0.2,
        entropy_coef:  float = 0.01,
        value_coef:    float = 0.5,
        n_steps:       int   = 512,
        n_epochs:      int   = 4,
        batch_size:    int   = 64,
        max_grad_norm: float = 0.5,
        device:        str   = "auto",
        checkpoint_dir: str  = "./checkpoints",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch not installed")

        self.env    = env
        self.gamma  = gamma
        self.gae    = gae_lambda
        self.clip   = clip_eps
        self.ent_c  = entropy_coef
        self.val_c  = value_coef
        self.nsteps = n_steps
        self.epochs = n_epochs
        self.bsize  = batch_size
        self.maxgn  = max_grad_norm

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = PokemonActorCritic().to(self.device)
        self.opt   = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.LinearLR(
            self.opt, start_factor=1.0, end_factor=0.1, total_iters=1000
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step   = 0
        self.best_reward   = -math.inf
        self.training_log: List[Dict] = []

    def _obs_to_tensors(self, obs: Dict) -> Tuple["torch.Tensor", "torch.Tensor"]:
        screen = torch.tensor(obs["screen"], dtype=torch.uint8).unsqueeze(0).to(self.device)
        gs     = torch.tensor(obs["game_state"], dtype=torch.float32).unsqueeze(0).to(self.device)
        return screen, gs

    def collect_rollout(self) -> Tuple[RolloutBuffer, float]:
        buf  = RolloutBuffer(self.nsteps)
        obs  = self.env.reset()
        hx   = cx = None
        ep_reward = 0.0

        for _ in range(self.nsteps):
            screen, gs = self._obs_to_tensors(obs)
            with torch.no_grad():
                action, log_p, value, hx, cx = self.model.act(screen, gs, hx, cx)

            a     = int(action.item())
            obs2, reward, done, info = self.env.step(a)

            buf.add(
                obs["screen"], obs["game_state"],
                a, float(log_p.item()),
                reward, float(value.item()), done,
            )
            ep_reward += reward
            obs = obs2
            self.global_step += 1

            if done:
                obs  = self.env.reset()
                hx   = cx = None

        # Bootstrap
        screen, gs = self._obs_to_tensors(obs)
        with torch.no_grad():
            _, last_val, _, _ = self.model(screen, gs, hx, cx)

        return buf, ep_reward, float(last_val.item())

    def update(self, buf: RolloutBuffer, last_val: float) -> Dict[str, float]:
        returns, advantages = buf.compute_returns(last_val, self.gamma, self.gae)

        # Normalise advantages
        adv = np.array(advantages, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        screens     = np.array(buf.screens,     dtype=np.uint8)
        game_states = np.array(buf.game_states, dtype=np.float32)
        actions     = np.array(buf.actions,     dtype=np.int64)
        old_lps     = np.array(buf.log_probs,   dtype=np.float32)
        rets        = np.array(returns,         dtype=np.float32)

        N          = len(actions)
        total_loss = pg_loss = vf_loss = ent_loss = 0.0

        for _ in range(self.epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, self.bsize):
                mb = idx[start:start + self.bsize]

                s_sc  = torch.tensor(screens[mb],     dtype=torch.uint8).to(self.device)
                s_gs  = torch.tensor(game_states[mb], dtype=torch.float32).to(self.device)
                s_act = torch.tensor(actions[mb],     dtype=torch.long).to(self.device)
                s_olp = torch.tensor(old_lps[mb],     dtype=torch.float32).to(self.device)
                s_ret = torch.tensor(rets[mb],        dtype=torch.float32).to(self.device)
                s_adv = torch.tensor(adv[mb],         dtype=torch.float32).to(self.device)

                logits, values, _, _ = self.model(s_sc, s_gs)
                dist  = Categorical(logits=logits)
                lp    = dist.log_prob(s_act)
                ent   = dist.entropy().mean()

                ratio    = torch.exp(lp - s_olp)
                pg1      = ratio * s_adv
                pg2      = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * s_adv
                p_loss   = -torch.min(pg1, pg2).mean()
                v_loss   = F.mse_loss(values, s_ret)

                loss = p_loss + self.val_c * v_loss - self.ent_c * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.maxgn)
                self.opt.step()

                total_loss += loss.item()
                pg_loss    += p_loss.item()
                vf_loss    += v_loss.item()
                ent_loss   += ent.item()

        self.sched.step()
        nb = max(1, self.epochs * (N // self.bsize))
        return {
            "loss":     total_loss / nb,
            "pg_loss":  pg_loss    / nb,
            "vf_loss":  vf_loss    / nb,
            "entropy":  ent_loss   / nb,
        }

    def train(self, total_steps: int = 50_000, log_interval: int = 10) -> List[Dict]:
        iteration = 0
        while self.global_step < total_steps:
            t0 = time.time()
            buf, ep_reward, last_val = self.collect_rollout()
            metrics = self.update(buf, last_val)
            dt = time.time() - t0

            metrics.update({
                "iteration":    iteration,
                "global_step":  self.global_step,
                "ep_reward":    ep_reward,
                "steps_per_sec": self.nsteps / dt,
            })
            self.training_log.append(metrics)

            if iteration % log_interval == 0:
                print(
                    f"[{iteration:4d}] step={self.global_step:7d} "
                    f"rew={ep_reward:.3f} loss={metrics['loss']:.4f} "
                    f"ent={metrics['entropy']:.4f} "
                    f"sps={metrics['steps_per_sec']:.0f}"
                )

            # Save best checkpoint
            if ep_reward > self.best_reward:
                self.best_reward = ep_reward
                self.save_checkpoint("best")

            # Periodic checkpoint
            if iteration % 50 == 0:
                self.save_checkpoint(f"iter_{iteration:05d}")

            iteration += 1

        return self.training_log

    def save_checkpoint(self, name: str):
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save({
            "model": self.model.state_dict(),
            "opt":   self.opt.state_dict(),
            "step":  self.global_step,
            "best":  self.best_reward,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.global_step  = ckpt.get("step", 0)
        self.best_reward  = ckpt.get("best", -math.inf)
        print(f"Loaded checkpoint from {path} (step={self.global_step})")
