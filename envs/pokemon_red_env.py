"""
Pokemon Red RL Environment — OpenEnv Spec Compliant
Wraps PyBoy + custom reward logic into a step()/reset()/state() interface.
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from pyboy import PyBoy
    from pyboy.utils import WindowEvent
    PYBOY_AVAILABLE = True
except ImportError:
    PYBOY_AVAILABLE = False

# ── Memory addresses (Pokemon Red / Blue offsets) ──────────────────────────
MEM = {
    # Party
    "party_size":        0xD163,
    "party_mon1_hp":     0xD16C,
    "party_mon1_maxhp":  0xD18D,
    "party_mon1_level":  0xD18C,
    "party_mon1_species":0xD164,
    # Badges
    "badges":            0xD356,
    # Player position
    "map_id":            0xD35E,
    "x_pos":             0xD362,
    "y_pos":             0xD361,
    # Events / flags
    "events_base":       0xD747,
    # Money
    "money_hi":          0xD347,
    "money_mid":         0xD348,
    "money_lo":          0xD349,
    # Seen / owned pokemon
    "pokemon_seen":      0xD30A,
    "pokemon_owned":     0xD2F7,
}

if PYBOY_AVAILABLE:
    ACTION_MAP = {
        0: WindowEvent.PRESS_ARROW_DOWN,
        1: WindowEvent.PRESS_ARROW_LEFT,
        2: WindowEvent.PRESS_ARROW_RIGHT,
        3: WindowEvent.PRESS_ARROW_UP,
        4: WindowEvent.PRESS_BUTTON_A,
        5: WindowEvent.PRESS_BUTTON_B,
        6: WindowEvent.PRESS_BUTTON_START,
        7: WindowEvent.PRESS_BUTTON_SELECT,
    }
    RELEASE_MAP = {
        0: WindowEvent.RELEASE_ARROW_DOWN,
        1: WindowEvent.RELEASE_ARROW_LEFT,
        2: WindowEvent.RELEASE_ARROW_RIGHT,
        3: WindowEvent.RELEASE_ARROW_UP,
        4: WindowEvent.RELEASE_BUTTON_A,
        5: WindowEvent.RELEASE_BUTTON_B,
        6: WindowEvent.RELEASE_BUTTON_START,
        7: WindowEvent.RELEASE_BUTTON_SELECT,
    }
else:
    # Stub placeholders — only used by PokemonRedEnv, not PokemonRedStubEnv
    ACTION_MAP  = {i: i for i in range(8)}
    RELEASE_MAP = {i: i for i in range(8)}


class PokemonRedEnv:
    """
    OpenEnv-compatible Mini-RL environment for Pokemon Red.

    Required env vars:
        ROM_PATH   — path to pokemon_red.gb (not distributed)
        SAVE_PATH  — optional directory for save states
    """

    # ── OpenEnv metadata ──────────────────────────────────────────────────
    metadata: Dict[str, Any] = {
        "name": "PokemonRed-v1",
        "version": "1.0.0",
        "action_space": {"type": "discrete", "n": 8},
        "observation_space": {
            "type": "dict",
            "spaces": {
                "screen":       {"type": "box", "shape": [144, 160, 3], "dtype": "uint8"},
                "game_state":   {"type": "box", "shape": [32],          "dtype": "float32"},
            },
        },
        "tasks": [
            "obtain_first_badge",
            "reach_cerulean_city",
            "catch_pokemon",
            "defeat_rival_oak_lab",
            "level_up_starter",
        ],
        "reward_range": [0.0, 1.0],
    }

    # ── Init ──────────────────────────────────────────────────────────────
    def __init__(
        self,
        rom_path: Optional[str] = None,
        save_path: Optional[str] = None,
        max_steps: int = 2048,
        headless: bool = True,
        frame_skip: int = 24,
        task: str = "obtain_first_badge",
    ):
        self.rom_path   = rom_path or os.environ.get("ROM_PATH", "pokemon_red.gb")
        self.save_path  = save_path or os.environ.get("SAVE_PATH", "./saves")
        self.max_steps  = max_steps
        self.headless   = headless
        self.frame_skip = frame_skip
        self.task       = task

        self.pyboy: Optional[PyBoy] = None
        self._step_count   = 0
        self._total_reward = 0.0
        self._visited_maps: set  = set()
        self._visited_tiles: set = set()
        self._badges_seen  = 0
        self._max_level    = 1
        self._pokemon_seen = 0
        self._screenshot_buf: Optional[np.ndarray] = None

        Path(self.save_path).mkdir(parents=True, exist_ok=True)

    # ── PyBoy helpers ─────────────────────────────────────────────────────
    def _init_pyboy(self):
        if not PYBOY_AVAILABLE:
            raise RuntimeError("pyboy not installed. Run: pip install pyboy")
        window = "null" if self.headless else "SDL2"
        self.pyboy = PyBoy(
            self.rom_path,
            window_type=window,
            window_scale=1,
            debug=False,
            game_wrapper=True,
        )
        self.pyboy.set_emulation_speed(0)  # max speed

    def _read_mem(self, addr: int) -> int:
        return self.pyboy.get_memory_value(addr)

    def _read_bcd(self, *addrs: int) -> int:
        """Read BCD-encoded multi-byte value (e.g. money)."""
        val = 0
        for a in addrs:
            b = self._read_mem(a)
            val = val * 100 + (b >> 4) * 10 + (b & 0x0F)
        return val

    def _count_bits(self, base: int, n_bytes: int) -> int:
        total = 0
        for i in range(n_bytes):
            total += bin(self._read_mem(base + i)).count("1")
        return total

    def _get_screen(self) -> np.ndarray:
        screen = self.pyboy.botsupport_manager().screen()
        return np.array(screen.screen_ndarray(), dtype=np.uint8)  # (144,160,3)

    def _press_action(self, action: int):
        self.pyboy.send_input(ACTION_MAP[action])
        for _ in range(self.frame_skip):
            self.pyboy.tick()
        self.pyboy.send_input(RELEASE_MAP[action])
        self.pyboy.tick()

    # ── Game state extraction ─────────────────────────────────────────────
    def _game_state_vector(self) -> np.ndarray:
        badges   = self._read_mem(MEM["badges"])
        n_badges = bin(badges).count("1")
        level    = self._read_mem(MEM["party_mon1_level"])
        hp       = self._read_mem(MEM["party_mon1_hp"])
        maxhp    = max(self._read_mem(MEM["party_mon1_maxhp"]), 1)
        map_id   = self._read_mem(MEM["map_id"])
        x_pos    = self._read_mem(MEM["x_pos"])
        y_pos    = self._read_mem(MEM["y_pos"])
        p_size   = self._read_mem(MEM["party_size"])
        n_seen   = self._count_bits(MEM["pokemon_seen"],  19)
        n_owned  = self._count_bits(MEM["pokemon_owned"], 19)

        vec = np.zeros(32, dtype=np.float32)
        vec[0]  = n_badges   / 8.0
        vec[1]  = level      / 100.0
        vec[2]  = hp         / max(maxhp, 1)
        vec[3]  = map_id     / 255.0
        vec[4]  = x_pos      / 255.0
        vec[5]  = y_pos      / 255.0
        vec[6]  = p_size     / 6.0
        vec[7]  = n_seen     / 151.0
        vec[8]  = n_owned    / 151.0
        vec[9]  = len(self._visited_tiles) / 2000.0
        vec[10] = self._step_count / self.max_steps
        return vec

    # ── Reward shaping ────────────────────────────────────────────────────
    def _compute_reward(self) -> float:
        reward = 0.0

        # 1. Exploration reward
        map_id = self._read_mem(MEM["map_id"])
        x_pos  = self._read_mem(MEM["x_pos"])
        y_pos  = self._read_mem(MEM["y_pos"])
        tile_key = (map_id, x_pos, y_pos)
        if tile_key not in self._visited_tiles:
            self._visited_tiles.add(tile_key)
            reward += 0.01
        if map_id not in self._visited_maps:
            self._visited_maps.add(map_id)
            reward += 0.5

        # 2. Badge reward
        badges   = self._read_mem(MEM["badges"])
        n_badges = bin(badges).count("1")
        if n_badges > self._badges_seen:
            reward += 2.0 * (n_badges - self._badges_seen)
            self._badges_seen = n_badges

        # 3. Level-up reward
        level = self._read_mem(MEM["party_mon1_level"])
        if level > self._max_level:
            reward += 0.3 * (level - self._max_level)
            self._max_level = level

        # 4. Pokémon seen reward
        n_seen = self._count_bits(MEM["pokemon_seen"], 19)
        if n_seen > self._pokemon_seen:
            reward += 0.2 * (n_seen - self._pokemon_seen)
            self._pokemon_seen = n_seen

        return float(np.clip(reward, 0.0, 5.0))

    def _is_done(self) -> bool:
        if self._step_count >= self.max_steps:
            return True
        # All 8 badges → episode complete
        badges = self._read_mem(MEM["badges"])
        if bin(badges).count("1") >= 8:
            return True
        return False

    # ── OpenEnv interface ─────────────────────────────────────────────────
    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return initial observation."""
        if self.pyboy is None:
            self._init_pyboy()
        else:
            self.pyboy.stop(save=False)
            self._init_pyboy()

        # Skip BIOS / title screen
        for _ in range(500):
            self.pyboy.tick()

        self._step_count   = 0
        self._total_reward = 0.0
        self._visited_maps  = set()
        self._visited_tiles = set()
        self._badges_seen   = 0
        self._max_level     = 1
        self._pokemon_seen  = 0

        return self._get_obs()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Execute one action. Returns (obs, reward, done, info)."""
        if self.pyboy is None:
            raise RuntimeError("Call reset() before step().")
        if action not in ACTION_MAP:
            raise ValueError(f"Invalid action {action}. Must be 0-7.")

        self._press_action(action)
        self._step_count += 1

        obs    = self._get_obs()
        reward = self._compute_reward()
        done   = self._is_done()

        self._total_reward += reward

        info = {
            "step":          self._step_count,
            "total_reward":  self._total_reward,
            "badges":        self._badges_seen,
            "level":         self._max_level,
            "maps_visited":  len(self._visited_maps),
            "tiles_visited": len(self._visited_tiles),
        }
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the current state."""
        if self.pyboy is None:
            return {"status": "not_initialized"}
        gs = self._game_state_vector()
        return {
            "step":          self._step_count,
            "total_reward":  self._total_reward,
            "badges":        self._badges_seen,
            "level":         self._max_level,
            "maps_visited":  len(self._visited_maps),
            "tiles_visited": len(self._visited_tiles),
            "game_state":    gs.tolist(),
        }

    def _get_obs(self) -> Dict[str, Any]:
        screen = self._get_screen()
        gs     = self._game_state_vector()
        return {"screen": screen, "game_state": gs}

    def close(self):
        if self.pyboy:
            self.pyboy.stop(save=False)
            self.pyboy = None

    def __del__(self):
        self.close()


# ── Stub env (no ROM required — for CI / grader testing) ──────────────────
class PokemonRedStubEnv(PokemonRedEnv):
    """
    Drop-in stub that fakes the emulator for unit tests and CI.
    Returns random pixel frames and incrementing game-state vectors.
    """

    def __init__(self, task: str = "obtain_first_badge", max_steps: int = 100):
        self.task        = task
        self.max_steps   = max_steps
        self.pyboy       = True  # non-None sentinel
        self._step_count = 0
        self._total_reward = 0.0
        self._visited_maps  = set()
        self._visited_tiles = set()
        self._badges_seen   = 0
        self._max_level     = 1
        self._pokemon_seen  = 0
        self._fake_state = {
            "badges":   0,
            "level":    5,
            "map_id":   1,
            "x_pos":    10,
            "y_pos":    10,
            "party_size": 1,
            "seen":     3,
            "owned":    1,
        }

    def _get_screen(self) -> np.ndarray:
        rng = np.random.default_rng(self._step_count)
        return rng.integers(0, 255, (144, 160, 3), dtype=np.uint8)

    def _press_action(self, action: int):
        # Simulate movement / exploration
        fs = self._fake_state
        if action == 0: fs["y_pos"] = min(255, fs["y_pos"] + 1)
        elif action == 1: fs["x_pos"] = max(0, fs["x_pos"] - 1)
        elif action == 2: fs["x_pos"] = min(255, fs["x_pos"] + 1)
        elif action == 3: fs["y_pos"] = max(0, fs["y_pos"] - 1)
        # Occasionally gain a level or badge
        if self._step_count % 20 == 0:
            fs["level"] = min(100, fs["level"] + 1)
        if self._step_count % 50 == 0 and fs["badges"] < 8:
            fs["badges"] += 1

    def _read_mem(self, addr: int) -> int:
        fs = self._fake_state
        return {
            MEM["badges"]:            fs["badges"],
            MEM["party_mon1_level"]:  fs["level"],
            MEM["party_mon1_hp"]:     50,
            MEM["party_mon1_maxhp"]:  55,
            MEM["map_id"]:            fs["map_id"],
            MEM["x_pos"]:             fs["x_pos"],
            MEM["y_pos"]:             fs["y_pos"],
            MEM["party_size"]:        fs["party_size"],
        }.get(addr, 0)

    def _count_bits(self, base: int, n_bytes: int) -> int:
        if base == MEM["pokemon_seen"]:  return self._fake_state["seen"]
        if base == MEM["pokemon_owned"]: return self._fake_state["owned"]
        return 0

    def _init_pyboy(self): pass

    def reset(self) -> Dict[str, Any]:
        self._step_count   = 0
        self._total_reward = 0.0
        self._visited_maps  = set()
        self._visited_tiles = set()
        self._badges_seen   = 0
        self._max_level     = 5
        self._pokemon_seen  = 0
        self._fake_state = {"badges":0,"level":5,"map_id":1,"x_pos":10,"y_pos":10,"party_size":1,"seen":3,"owned":1}
        return self._get_obs()

    def close(self): pass
