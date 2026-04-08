"""
Task definitions + graders for the Pokemon Red RL hackathon environment.
Each task must return a score in [0.0, 1.0].
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional


# ── Base grader ────────────────────────────────────────────────────────────
class BaseGrader:
    task_id: str = "base"
    description: str = ""

    def grade(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Score a completed episode trajectory.
        trajectory: list of step dicts, each containing:
            {"obs": ..., "action": int, "reward": float, "done": bool, "info": dict}
        Returns float in [0.0, 1.0].
        """
        raise NotImplementedError

    def _last_info(self, trajectory: List[Dict]) -> Dict:
        for step in reversed(trajectory):
            if step.get("info"):
                return step["info"]
        return {}

    def _max_info_key(self, trajectory: List[Dict], key: str) -> float:
        vals = [s["info"].get(key, 0) for s in trajectory if s.get("info")]
        return max(vals) if vals else 0.0


# ── Task 1: Obtain First Badge ────────────────────────────────────────────
class ObtainFirstBadgeGrader(BaseGrader):
    task_id     = "obtain_first_badge"
    description = "Agent must defeat Brock and earn the Boulder Badge."

    def grade(self, trajectory: List[Dict]) -> float:
        max_badges = self._max_info_key(trajectory, "badges")
        if max_badges >= 1:
            # Bonus for reaching it quickly
            total_steps = len(trajectory)
            speed_bonus = max(0.0, 1.0 - total_steps / 2048.0) * 0.2
            return min(1.0, 0.8 + speed_bonus)
        # Partial credit: exploration
        tiles = self._max_info_key(trajectory, "tiles_visited")
        return min(0.79, tiles / 500.0)


# ── Task 2: Reach Cerulean City ───────────────────────────────────────────
CERULEAN_MAP_ID = 3   # Pokemon Red map ID for Cerulean City

class ReachCeruleanCityGrader(BaseGrader):
    task_id     = "reach_cerulean_city"
    description = "Agent must navigate to Cerulean City (past Mt. Moon)."

    def grade(self, trajectory: List[Dict]) -> float:
        # Check if Cerulean map was visited
        for step in trajectory:
            obs = step.get("obs", {})
            gs  = obs.get("game_state", [])
            if len(gs) >= 4:
                # gs[3] = map_id / 255
                if abs(gs[3] - CERULEAN_MAP_ID / 255.0) < 0.01:
                    return 1.0
        # Partial: how many maps explored
        maps = self._max_info_key(trajectory, "maps_visited")
        return min(0.9, maps / 10.0)


# ── Task 3: Catch a Pokémon ───────────────────────────────────────────────
class CatchPokemonGrader(BaseGrader):
    task_id     = "catch_pokemon"
    description = "Agent must catch at least one wild Pokémon."

    def grade(self, trajectory: List[Dict]) -> float:
        # Owned pokémon count increases when you catch one
        # gs[8] = n_owned / 151
        max_owned = max(
            (s["obs"]["game_state"][8] for s in trajectory
             if s.get("obs") and len(s["obs"].get("game_state", [])) > 8),
            default=0.0
        )
        if max_owned > 0:
            # At least 1/151 means a catch happened
            owned_count = round(max_owned * 151)
            return min(1.0, 0.6 + owned_count * 0.1)
        return 0.0


# ── Task 4: Defeat Rival (Oak's Lab) ─────────────────────────────────────
OAK_LAB_MAP_ID = 40

class DefeatRivalOakLabGrader(BaseGrader):
    task_id     = "defeat_rival_oak_lab"
    description = "Agent must win the opening rival battle in Oak's Lab."

    def grade(self, trajectory: List[Dict]) -> float:
        # Heuristic: if agent visited Oak's lab map AND level increased
        visited_lab = False
        max_level   = 1
        for step in trajectory:
            obs = step.get("obs", {})
            gs  = obs.get("game_state", [])
            if len(gs) >= 4:
                if abs(gs[3] - OAK_LAB_MAP_ID / 255.0) < 0.01:
                    visited_lab = True
            if len(gs) >= 2:
                lvl = round(gs[1] * 100)
                max_level = max(max_level, lvl)

        if visited_lab and max_level >= 6:
            return 1.0
        elif visited_lab:
            return 0.5
        return 0.0


# ── Task 5: Level Up Starter ──────────────────────────────────────────────
class LevelUpStarterGrader(BaseGrader):
    task_id     = "level_up_starter"
    description = "Agent's starter Pokémon must reach at least Level 10."

    TARGET_LEVEL = 10

    def grade(self, trajectory: List[Dict]) -> float:
        max_level = 1
        for step in trajectory:
            obs = step.get("obs", {})
            gs  = obs.get("game_state", [])
            if len(gs) >= 2:
                lvl = round(gs[1] * 100)
                max_level = max(max_level, lvl)

        if max_level >= self.TARGET_LEVEL:
            # Bonus for exceeding target
            excess = max_level - self.TARGET_LEVEL
            return min(1.0, 0.8 + excess * 0.02)
        return max(0.0, max_level / self.TARGET_LEVEL * 0.79)


# ── Registry ──────────────────────────────────────────────────────────────
GRADER_REGISTRY: Dict[str, BaseGrader] = {
    g.task_id: g()
    for g in [
        ObtainFirstBadgeGrader,
        ReachCeruleanCityGrader,
        CatchPokemonGrader,
        DefeatRivalOakLabGrader,
        LevelUpStarterGrader,
    ]
}


def get_grader(task_id: str) -> BaseGrader:
    if task_id not in GRADER_REGISTRY:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(GRADER_REGISTRY)}")
    return GRADER_REGISTRY[task_id]


def run_all_graders(trajectory: List[Dict]) -> Dict[str, float]:
    """Run every grader on a trajectory. Returns task_id -> score mapping."""
    return {tid: g.grade(trajectory) for tid, g in GRADER_REGISTRY.items()}
