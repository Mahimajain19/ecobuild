"""
Indian Grid Model for EcoBuild.
Simulates power cuts, load shedding schedules, and voltage fluctuations —
critical realities for 60%+ of Indian tier-2/3 cities.

Grid reliability by tier:
  Tier-1 (Delhi/Mumbai/Bangalore): 0-2 hrs cuts/day
  Tier-2 (Lucknow/Patna/Bhubaneswar): 4-8 hrs/day
  Tier-3 (Rural): 8-12 hrs/day
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


CITY_PROFILES = {
    "tier1": {"cut_hours_per_day": 1.0, "voltage_drop_prob": 0.05},
    "tier2": {"cut_hours_per_day": 5.0, "voltage_drop_prob": 0.20},
    "tier3": {"cut_hours_per_day": 10.0, "voltage_drop_prob": 0.40},
    "delhi": {"cut_hours_per_day": 0.5, "voltage_drop_prob": 0.03},
    "mumbai": {"cut_hours_per_day": 0.3, "voltage_drop_prob": 0.02},
    "bangalore": {"cut_hours_per_day": 1.0, "voltage_drop_prob": 0.05},
    "lucknow": {"cut_hours_per_day": 5.0, "voltage_drop_prob": 0.20},
    "patna": {"cut_hours_per_day": 8.0, "voltage_drop_prob": 0.30},
}

SAFE_VOLTAGE_MIN = 200.0
SAFE_VOLTAGE_MAX = 240.0
BROWNOUT_VOLTAGE = 180.0
DAMAGE_VOLTAGE = 160.0
NOMINAL_VOLTAGE = 220.0


@dataclass
class GridState:
    is_available: bool
    voltage: float
    minutes_since_last_cut: int
    predicted_next_cut_minutes: int
    is_brownout: bool
    is_damage_zone: bool


class IndianGridModel:
    """
    Simulates Indian power grid with scheduled/stochastic outages and voltage fluctuations.
    All randomness is seeded for reproducibility.
    """

    def __init__(
        self,
        city: str = "tier2",
        mode: str = "scheduled",
        rng: Optional[np.random.RandomState] = None,
        total_steps: int = 144,
    ):
        """
        Args:
            city: City/tier profile name
            mode: "scheduled" (deterministic cuts) or "stochastic" (random)
            rng: Seeded random state
            total_steps: Episode length in steps (for schedule generation)
        """
        self.city = city
        self.mode = mode
        self.rng = rng or np.random.RandomState(0)
        self.total_steps = total_steps

        profile = CITY_PROFILES.get(city, CITY_PROFILES["tier2"])
        self.cut_hours_per_day = profile["cut_hours_per_day"]
        self.voltage_drop_prob = profile["voltage_drop_prob"]

        # State
        self._cut_schedule: List[bool] = []  # True = grid available
        self._current_step = 0
        self._minutes_since_last_cut = 999
        self.total_cut_steps = 0

        self._generate_schedule()

    def reset(self, rng: Optional[np.random.RandomState] = None, total_steps: int = None):
        if rng is not None:
            self.rng = rng
        if total_steps is not None:
            self.total_steps = total_steps
        self._current_step = 0
        self._minutes_since_last_cut = 999
        self.total_cut_steps = 0
        self._generate_schedule()

    def _generate_schedule(self):
        """Generate a power cut schedule for the episode."""
        n_steps = self.total_steps
        schedule = [True] * n_steps  # Start: all available

        steps_per_hour = 6  # 10-min steps
        cut_steps = int(self.cut_hours_per_day * steps_per_hour * (n_steps / 144))

        if self.mode == "scheduled":
            # Deterministic: cuts happen at predictable hours (e.g., 2PM-4PM, 8PM-10PM)
            # Use 2 cut blocks with slight jitter
            if cut_steps > 0:
                block1_start = 84 + self.rng.randint(-6, 7)  # around 2PM ± 1hr
                block1_len = cut_steps // 2
                block2_start = 120 + self.rng.randint(-6, 7)  # around 8PM ± 1hr
                block2_len = cut_steps - block1_len

                for i in range(block1_start, min(block1_start + block1_len, n_steps)):
                    schedule[i] = False
                for i in range(block2_start, min(block2_start + block2_len, n_steps)):
                    schedule[i] = False

        elif self.mode == "stochastic":
            # Random: Poisson-distributed outages
            cut_indices = sorted(self.rng.choice(n_steps, size=min(cut_steps, n_steps), replace=False))
            for idx in cut_indices:
                schedule[idx] = False

        self._cut_schedule = schedule

    def is_grid_available(self, step: int = None) -> bool:
        """Returns True if grid power is available at this step."""
        s = step if step is not None else self._current_step
        if s >= len(self._cut_schedule):
            return True
        return self._cut_schedule[s]

    def get_voltage(self, hour: int, season: str = "summer") -> float:
        """
        Returns current grid voltage. Sags during peak demand hours (afternoons).
        Season affects: summer peak demand is worst.
        """
        # Voltage sag: worst 2PM-8PM in summer, moderate other seasons
        season_factor = {"summer": 1.0, "monsoon": 0.6, "post_monsoon": 0.7, "winter": 0.5}.get(season, 0.7)

        if 14 <= hour < 20:
            sag = self.rng.uniform(0, 40.0) * season_factor * self.voltage_drop_prob * 10
        else:
            sag = self.rng.uniform(0, 15.0) * self.voltage_drop_prob * 10

        return float(np.clip(NOMINAL_VOLTAGE - sag, DAMAGE_VOLTAGE, 260.0))

    def step(self, hour: int, season: str = "summer") -> GridState:
        """Advance one time step and return grid state."""
        available = self.is_grid_available(self._current_step)

        if not available:
            self.total_cut_steps += 1
            self._minutes_since_last_cut = 0
        else:
            self._minutes_since_last_cut = min(999, self._minutes_since_last_cut + 10)

        voltage = self.get_voltage(hour, season) if available else 0.0

        # Predict next cut
        next_cut = self._find_next_cut(self._current_step)

        self._current_step += 1

        return GridState(
            is_available=available,
            voltage=voltage,
            minutes_since_last_cut=self._minutes_since_last_cut,
            predicted_next_cut_minutes=next_cut * 10,
            is_brownout=BROWNOUT_VOLTAGE <= voltage < SAFE_VOLTAGE_MIN and available,
            is_damage_zone=voltage < BROWNOUT_VOLTAGE and available,
        )

    def _find_next_cut(self, current_step: int) -> int:
        """Returns steps until next cut (or 999 if none scheduled)."""
        for i in range(current_step + 1, len(self._cut_schedule)):
            if not self._cut_schedule[i]:
                return i - current_step
        return 999

    def equipment_damage_penalty(self, voltage: float, compressor_on: bool) -> float:
        """Returns penalty for running AC compressor at unsafe voltage."""
        if not compressor_on:
            return 0.0
        if voltage < DAMAGE_VOLTAGE:
            return 5.0  # Severe: equipment damage risk
        elif voltage < BROWNOUT_VOLTAGE:
            return 1.5  # Moderate: efficiency loss
        return 0.0
