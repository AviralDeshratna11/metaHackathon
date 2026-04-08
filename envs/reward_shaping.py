"""
envs/reward_shaping.py — Reward Shaping & Curriculum Manager
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Centralises all reward logic so it can be tuned independently
of the environment wrapper. Also implements curriculum learning:
task difficulty ramps up as the agent improves.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# ── Pokémon Red milestone map ─────────────────────────────────────────────
# Approximate Game Boy map IDs for key locations
MILESTONE_MAPS = {
    1:  ("Pallet Town",         0.00),
    2:  ("Route 1",             0.05),
    4:  ("Viridian City",       0.10),
    5:  ("Route 2",             0.13),
    6:  ("Viridian Forest",     0.18),
    7:  ("Route 3 (Pewter)",    0.22),
    8:  ("Pewter City",         0.28),
    9:  ("Route 4",             0.32),
    11: ("Mt. Moon entrance",   0.40),
    12: ("Mt. Moon deep",       0.48),
    3:  ("Cerulean City",       0.60),
    16: ("Route 5",             0.65),
    6:  ("Vermilion City",      0.75),
}

BADGE_REWARD        = 2.0    # per badge
LEVEL_UP_REWARD     = 0.25   # per level gained
NEW_MAP_REWARD      = 0.5    # first visit to each map
NEW_TILE_REWARD     = 0.01   # first visit to each tile
POKEMON_SEEN_REWARD = 0.15   # per new pokémon seen
POKEMON_CAUGHT_REWARD = 0.30 # per pokémon caught
HEAL_REWARD         = 0.05   # when HP is restored (visited Pokémon Center)
KO_PENALTY          = -0.10  # when party is wiped

# Milestone bonus: reward for reaching each key location for the first time
MILESTONE_BONUS     = 1.0


class RewardShaper:
    """
    Stateful reward augmenter. Attach one instance per environment.
    Call compute(obs, info) each step to get the shaped reward delta.
    """

    def __init__(
        self,
        badge_r:     float = BADGE_REWARD,
        levelup_r:   float = LEVEL_UP_REWARD,
        new_map_r:   float = NEW_MAP_REWARD,
        new_tile_r:  float = NEW_TILE_REWARD,
        seen_r:      float = POKEMON_SEEN_REWARD,
        caught_r:    float = POKEMON_CAUGHT_REWARD,
        milestone_r: float = MILESTONE_BONUS,
        ko_penalty:  float = KO_PENALTY,
    ):
        self.badge_r     = badge_r
        self.levelup_r   = levelup_r
        self.new_map_r   = new_map_r
        self.new_tile_r  = new_tile_r
        self.seen_r      = seen_r
        self.caught_r    = caught_r
        self.milestone_r = milestone_r
        self.ko_penalty  = ko_penalty
        self.reset()

    def reset(self):
        self._badges       = 0
        self._level        = 1
        self._seen         = 0
        self._owned        = 0
        self._visited_maps: set  = set()
        self._visited_tiles: set = set()
        self._milestones_hit: set = set()
        self._prev_hp_ratio = 1.0

    def compute(self, obs: Dict, info: Dict, env_state: Dict) -> float:
        """Return total shaped reward delta for this step."""
        reward = 0.0
        gs = obs.get("game_state", np.zeros(32, dtype=np.float32))

        # ── Decode game state ──────────────────────────────────────────
        hp_ratio   = float(gs[2]) if len(gs) > 2 else 1.0
        map_id     = round(float(gs[3]) * 255) if len(gs) > 3 else 0
        x          = round(float(gs[4]) * 255) if len(gs) > 4 else 0
        y          = round(float(gs[5]) * 255) if len(gs) > 5 else 0
        level      = round(float(gs[1]) * 100) if len(gs) > 1 else 1
        n_seen     = round(float(gs[7]) * 151) if len(gs) > 7 else 0
        n_owned    = round(float(gs[8]) * 151) if len(gs) > 8 else 0
        badges     = info.get("badges", self._badges)

        # ── Badge reward ──────────────────────────────────────────────
        if badges > self._badges:
            reward += self.badge_r * (badges - self._badges)
            self._badges = badges

        # ── Level-up reward ───────────────────────────────────────────
        if level > self._level:
            reward += self.levelup_r * (level - self._level)
            self._level = level

        # ── New map reward ────────────────────────────────────────────
        if map_id not in self._visited_maps:
            self._visited_maps.add(map_id)
            reward += self.new_map_r
            # Milestone bonus
            if map_id in MILESTONE_MAPS and map_id not in self._milestones_hit:
                self._milestones_hit.add(map_id)
                name, _ = MILESTONE_MAPS[map_id]
                reward += self.milestone_r
                print(f"  [Reward] Milestone reached: {name} (+{self.milestone_r:.2f})")

        # ── New tile reward ───────────────────────────────────────────
        tile = (map_id, x, y)
        if tile not in self._visited_tiles:
            self._visited_tiles.add(tile)
            reward += self.new_tile_r

        # ── Pokémon seen reward ───────────────────────────────────────
        if n_seen > self._seen:
            reward += self.seen_r * (n_seen - self._seen)
            self._seen = n_seen

        # ── Pokémon caught reward ─────────────────────────────────────
        if n_owned > self._owned:
            reward += self.caught_r * (n_owned - self._owned)
            self._owned = n_owned

        # ── KO penalty ───────────────────────────────────────────────
        if hp_ratio < 0.01 and self._prev_hp_ratio >= 0.01:
            reward += self.ko_penalty

        self._prev_hp_ratio = hp_ratio
        return float(np.clip(reward, -2.0, 5.0))

    @property
    def stats(self) -> Dict:
        return {
            "badges":          self._badges,
            "level":           self._level,
            "maps_visited":    len(self._visited_maps),
            "tiles_visited":   len(self._visited_tiles),
            "milestones":      len(self._milestones_hit),
            "pokemon_seen":    self._seen,
            "pokemon_owned":   self._owned,
        }


# ── Curriculum manager ────────────────────────────────────────────────────
class CurriculumManager:
    """
    Progressively increases task difficulty as the agent improves.

    Stages:
      0: Stay alive for 100 steps (survival)
      1: Reach Viridian City
      2: Beat Viridian Forest
      3: Defeat Brock (Pewter Gym)
      4: Navigate Mt. Moon
      5: Reach Cerulean City
      6: Full game completion
    """

    STAGES = [
        {"name": "Survive",          "task": "level_up_starter",      "threshold": 0.5},
        {"name": "Viridian City",    "task": "level_up_starter",      "threshold": 0.7},
        {"name": "Viridian Forest",  "task": "obtain_first_badge",    "threshold": 0.4},
        {"name": "Pewter Gym",       "task": "obtain_first_badge",    "threshold": 0.8},
        {"name": "Mt. Moon",         "task": "reach_cerulean_city",   "threshold": 0.4},
        {"name": "Cerulean City",    "task": "reach_cerulean_city",   "threshold": 0.9},
        {"name": "Full game",        "task": "reach_cerulean_city",   "threshold": 1.0},
    ]

    def __init__(self, window: int = 10):
        self.stage         = 0
        self.window        = window
        self._score_buffer: List[float] = []

    def record_score(self, score: float):
        self._score_buffer.append(score)
        if len(self._score_buffer) > self.window:
            self._score_buffer.pop(0)

        avg = sum(self._score_buffer) / len(self._score_buffer)
        threshold = self.STAGES[self.stage]["threshold"]

        if avg >= threshold and self.stage < len(self.STAGES) - 1:
            self.stage += 1
            next_stage = self.STAGES[self.stage]
            print(
                f"\n[Curriculum] ▶ Advancing to Stage {self.stage}: "
                f"{next_stage['name']} (task={next_stage['task']})\n"
            )

    @property
    def current_task(self) -> str:
        return self.STAGES[self.stage]["task"]

    @property
    def current_stage_name(self) -> str:
        return self.STAGES[self.stage]["name"]

    @property
    def progress(self) -> float:
        return self.stage / (len(self.STAGES) - 1)

    def summary(self) -> Dict:
        return {
            "stage":      self.stage,
            "name":       self.current_stage_name,
            "task":       self.current_task,
            "progress":   self.progress,
            "avg_score":  sum(self._score_buffer) / max(1, len(self._score_buffer)),
        }
