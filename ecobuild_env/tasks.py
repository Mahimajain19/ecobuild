"""
Graders for EcoBuild's 4 RL tasks.
Each grader is a class with documented weights, grade() returning [0.0, 1.0],
and grade_breakdown() for per-criterion transparency.

India-grid emission factor: 0.82 kg CO₂/kWh (CEA 2023 data)
"""

from dataclasses import dataclass
from typing import List, Optional


CO2_KG_PER_KWH = 0.82  # India grid emission factor

# Baselines per task
BASELINE_ALWAYS_ON_KWH_PER_STEP = (5.0 + 0.5 + 3.5) * (1/6)  # heater+lights+AC max
BASELINE_SMART_KWH_PER_STEP = 5.0 * (1/6)                       # heater only, always on


@dataclass
class StepData:
    """Data captured per step for grading."""
    energy_kwh: float
    energy_cost_inr: float
    occupied: bool
    occupancy_count: int
    temp: float
    heater_on: bool
    ac_on: bool
    lights_on: bool
    grid_available: bool
    genset_on: bool
    genset_cost_inr: float
    solar_kw: float
    humidity: float
    indoor_aqi: float


# ─────────────────────────────────────────────
# Task 1: basic_thermostat (EASY)
# ─────────────────────────────────────────────

class Task1Grader:
    """
    Task: Basic Thermostat Control (Delhi Winter, 24h)

    Grading formula:
      score = 0.40 × comfort_score + 0.40 × energy_efficiency + 0.20 × vacancy_score

    comfort_score   = % of occupied steps where temp is in [20, 22]°C
    energy_efficiency = max(0, 1 - energy_used / baseline_always_on)
    vacancy_score   = % of unoccupied steps where heater AND lights are off
    """
    WEIGHTS = {"comfort": 0.40, "energy_efficiency": 0.40, "vacancy_penalty": 0.20}
    COMFORT_MIN, COMFORT_MAX = 20.0, 22.0
    BASELINE_KWH = 5.0 * (1/6)  # Always-on heater baseline

    def grade(self, episode_data: List[StepData]) -> float:
        if not episode_data:
            return 0.0
        b = self.grade_breakdown(episode_data)
        return float(
            self.WEIGHTS["comfort"] * b["comfort_score"]
            + self.WEIGHTS["energy_efficiency"] * b["energy_efficiency"]
            + self.WEIGHTS["vacancy_penalty"] * b["vacancy_score"]
        )

    def grade_breakdown(self, episode_data: List[StepData]) -> dict:
        if not episode_data:
            return {"comfort_score": 0.0, "energy_efficiency": 0.0, "vacancy_score": 0.0}

        occupied_steps = [s for s in episode_data if s.occupied]
        unoccupied_steps = [s for s in episode_data if not s.occupied]

        comfort_score = 0.0
        if occupied_steps:
            in_range = sum(1 for s in occupied_steps if self.COMFORT_MIN <= s.temp <= self.COMFORT_MAX)
            comfort_score = in_range / len(occupied_steps)

        baseline_energy = self.BASELINE_KWH * len(episode_data)
        actual_energy = sum(s.energy_kwh for s in episode_data)
        energy_efficiency = max(0.0, 1.0 - actual_energy / max(baseline_energy, 1e-6))

        vacancy_score = 1.0
        if unoccupied_steps:
            wasted = sum(1 for s in unoccupied_steps if s.heater_on or s.lights_on)
            vacancy_score = max(0.0, 1.0 - wasted / len(unoccupied_steps))

        return {
            "comfort_score": round(comfort_score, 4),
            "energy_efficiency": round(energy_efficiency, 4),
            "vacancy_score": round(vacancy_score, 4),
        }


# ─────────────────────────────────────────────
# Task 2: day_night_tou (MEDIUM)
# ─────────────────────────────────────────────

class Task2Grader:
    """
    Task: Day/Night with Time-of-Use Tariffs (Delhi Summer, 24h)

    Grading formula:
      score = 0.30 × cost_efficiency + 0.30 × comfort_score
            + 0.25 × tou_awareness + 0.15 × vacancy_score

    cost_efficiency = max(0, 1 - total_cost / max_possible_cost)
    comfort_score   = % of occupied steps in [22, 25]°C
    tou_awareness   = fraction of heating/cooling shifted to off-peak hours
    vacancy_score   = % unoccupied steps with AC + lights off
    """
    WEIGHTS = {"cost": 0.30, "comfort": 0.30, "tou": 0.25, "vacancy": 0.15}
    COMFORT_MIN, COMFORT_MAX = 22.0, 25.0
    MAX_COST_PER_STEP = (5.0 + 3.5 + 0.5) * (1/6) * 8.0  # All on at peak rate

    def grade(self, episode_data: List[StepData]) -> float:
        if not episode_data:
            return 0.0
        b = self.grade_breakdown(episode_data)
        return float(
            self.WEIGHTS["cost"] * b["cost_efficiency"]
            + self.WEIGHTS["comfort"] * b["comfort_score"]
            + self.WEIGHTS["tou"] * b["tou_awareness"]
            + self.WEIGHTS["vacancy"] * b["vacancy_score"]
        )

    def grade_breakdown(self, episode_data: List[StepData]) -> dict:
        if not episode_data:
            return {k: 0.0 for k in ["cost_efficiency", "comfort_score", "tou_awareness", "vacancy_score"]}

        occupied_steps = [s for s in episode_data if s.occupied]
        unoccupied_steps = [s for s in episode_data if not s.occupied]

        total_cost = sum(s.energy_cost_inr for s in episode_data)
        max_cost = self.MAX_COST_PER_STEP * len(episode_data)
        cost_efficiency = max(0.0, 1.0 - total_cost / max(max_cost, 1e-6))

        comfort_score = 0.0
        if occupied_steps:
            in_range = sum(1 for s in occupied_steps if self.COMFORT_MIN <= s.temp <= self.COMFORT_MAX)
            comfort_score = in_range / len(occupied_steps)

        # TOU awareness: running HVAC heavily during off-peak (22:00-06:00) = good
        # Proxy: low cost relative to energy used = good TOU management
        total_energy = sum(s.energy_kwh for s in episode_data)
        avg_cost_per_kwh = total_cost / max(total_energy, 1e-6)
        tou_awareness = max(0.0, 1.0 - (avg_cost_per_kwh - 3.0) / 5.0)  # 3=best, 8=worst

        vacancy_score = 1.0
        if unoccupied_steps:
            wasted = sum(1 for s in unoccupied_steps if s.ac_on or s.lights_on)
            vacancy_score = max(0.0, 1.0 - wasted / len(unoccupied_steps))

        return {
            "cost_efficiency": round(cost_efficiency, 4),
            "comfort_score": round(comfort_score, 4),
            "tou_awareness": round(max(0.0, min(1.0, tou_awareness)), 4),
            "vacancy_score": round(vacancy_score, 4),
        }


# ─────────────────────────────────────────────
# Task 3: load_shedding_optimizer (MEDIUM-HARD)
# ─────────────────────────────────────────────

class Task3Grader:
    """
    Task: Load Shedding Optimizer (Mumbai Monsoon, 48h, 5h/day cuts)

    Grading formula:
      score = 0.40 × comfort_during_cuts + 0.30 × genset_cost_score + 0.30 × pre_conditioning_score

    comfort_during_cuts  = % of cut steps where temp stays in comfort zone
    genset_cost_score    = max(0, 1 - genset_cost_inr / max_expected_genset_cost)
    pre_conditioning     = fraction of cut-start events where temp was within 0.5°C of target beforehand
    """
    WEIGHTS = {"comfort_cuts": 0.40, "genset_cost": 0.30, "pre_cond": 0.30}
    COMFORT_MIN, COMFORT_MAX = 22.0, 26.0
    MAX_GENSET_COST_ESTIMATE = 500.0  # ₹ for 48h with heavy genset use

    def grade(self, episode_data: List[StepData]) -> float:
        if not episode_data:
            return 0.0
        b = self.grade_breakdown(episode_data)
        return float(
            self.WEIGHTS["comfort_cuts"] * b["comfort_during_cuts"]
            + self.WEIGHTS["genset_cost"] * b["genset_cost_score"]
            + self.WEIGHTS["pre_cond"] * b["pre_conditioning_score"]
        )

    def grade_breakdown(self, episode_data: List[StepData]) -> dict:
        if not episode_data:
            return {k: 0.0 for k in ["comfort_during_cuts", "genset_cost_score", "pre_conditioning_score"]}

        cut_steps = [s for s in episode_data if not s.grid_available]
        total_genset_cost = sum(s.genset_cost_inr for s in episode_data)

        comfort_during_cuts = 1.0
        if cut_steps:
            in_range = sum(1 for s in cut_steps if self.COMFORT_MIN <= s.temp <= self.COMFORT_MAX)
            comfort_during_cuts = in_range / len(cut_steps)

        genset_cost_score = max(0.0, 1.0 - total_genset_cost / self.MAX_GENSET_COST_ESTIMATE)

        # Pre-conditioning: temp within 0.5°C of midpoint before a cut starts
        target_mid = (self.COMFORT_MIN + self.COMFORT_MAX) / 2
        pre_cond_events = 0
        pre_cond_success = 0
        for i in range(1, len(episode_data)):
            if not episode_data[i].grid_available and episode_data[i - 1].grid_available:
                pre_cond_events += 1
                if abs(episode_data[i - 1].temp - target_mid) <= 0.5:
                    pre_cond_success += 1
        pre_cond_score = pre_cond_success / max(1, pre_cond_events)

        return {
            "comfort_during_cuts": round(comfort_during_cuts, 4),
            "genset_cost_score": round(genset_cost_score, 4),
            "pre_conditioning_score": round(pre_cond_score, 4),
        }


# ─────────────────────────────────────────────
# Task 4: multiday_optimization (HARD)
# ─────────────────────────────────────────────

class Task4Grader:
    """
    Task: Multi-Day Optimization (Delhi 7 days, Solar+Battery+TOU+AQI)

    Grading formula:
      score = 0.30 × cost_score + 0.30 × comfort_score + 0.20 × co2_score
            + 0.10 × anticipation_score + 0.10 × constraint_free_score

    cost_score       = max(0, 1 - total_cost / max_possible_cost_7days)
    comfort_score    = % of occupied time in [21, 25]°C
    co2_score        = max(0, 1 - total_co2_kg / baseline_co2_7days)
    anticipation     = pre-conditioning success rate before cuts + before high-occupancy events
    constraint_free  = fraction of steps with no extreme temp (<18°C or >28°C)
    """
    WEIGHTS = {"cost": 0.30, "comfort": 0.30, "co2": 0.20, "anticipation": 0.10, "constraint": 0.10}
    COMFORT_MIN, COMFORT_MAX = 21.0, 25.0
    CONSTRAINT_MIN, CONSTRAINT_MAX = 18.0, 28.0
    # 7-day max cost: all-on at peak rate
    MAX_COST_7DAYS = (5.0 + 3.5 + 0.5) * (1/6) * 8.0 * 1008
    # 7-day baseline CO2: average-on baseline
    BASELINE_CO2_KG = 4.0 * (1/6) * CO2_KG_PER_KWH * 1008

    def grade(self, episode_data: List[StepData]) -> float:
        if not episode_data:
            return 0.0
        b = self.grade_breakdown(episode_data)
        return float(
            self.WEIGHTS["cost"] * b["cost_score"]
            + self.WEIGHTS["comfort"] * b["comfort_score"]
            + self.WEIGHTS["co2"] * b["co2_score"]
            + self.WEIGHTS["anticipation"] * b["anticipation_score"]
            + self.WEIGHTS["constraint"] * b["constraint_free_score"]
        )

    def grade_breakdown(self, episode_data: List[StepData]) -> dict:
        if not episode_data:
            return {k: 0.0 for k in ["cost_score", "comfort_score", "co2_score", "anticipation_score", "constraint_free_score"]}

        occupied_steps = [s for s in episode_data if s.occupied]
        total_cost = sum(s.energy_cost_inr + s.genset_cost_inr for s in episode_data)
        total_energy_kwh = sum(s.energy_kwh for s in episode_data)

        cost_score = max(0.0, 1.0 - total_cost / max(self.MAX_COST_7DAYS, 1e-6))

        comfort_score = 0.0
        if occupied_steps:
            in_range = sum(1 for s in occupied_steps if self.COMFORT_MIN <= s.temp <= self.COMFORT_MAX)
            comfort_score = in_range / len(occupied_steps)

        total_co2 = total_energy_kwh * CO2_KG_PER_KWH
        co2_score = max(0.0, 1.0 - total_co2 / max(self.BASELINE_CO2_KG, 1e-6))

        # Anticipation: pre-heating before occupancy start
        anticipation_events = 0
        anticipation_success = 0
        for i in range(1, len(episode_data)):
            if episode_data[i].occupied and not episode_data[i - 1].occupied:
                anticipation_events += 1
                if episode_data[i - 1].heater_on or episode_data[i - 1].ac_on:
                    anticipation_success += 1
        anticipation_score = anticipation_success / max(1, anticipation_events)

        constraint_free = sum(
            1 for s in episode_data
            if self.CONSTRAINT_MIN <= s.temp <= self.CONSTRAINT_MAX
        ) / len(episode_data)

        return {
            "cost_score": round(cost_score, 4),
            "comfort_score": round(comfort_score, 4),
            "co2_score": round(co2_score, 4),
            "anticipation_score": round(anticipation_score, 4),
            "constraint_free_score": round(constraint_free, 4),
        }


# ─────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────

_GRADERS = {
    "basic_thermostat": Task1Grader,
    "day_night_tou": Task2Grader,
    "load_shedding_optimizer": Task3Grader,
    "multiday_optimization": Task4Grader,
    # Legacy alias
    "day_night_control": Task2Grader,
}


def evaluate_episode(task_name: str, episode_data: List[StepData]) -> float:
    """Evaluate an episode and return a score in [0.0, 1.0]."""
    cls = _GRADERS.get(task_name)
    if cls is None:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(_GRADERS.keys())}")
    return cls().grade(episode_data)


def evaluate_episode_breakdown(task_name: str, episode_data: List[StepData]) -> dict:
    """Evaluate and return per-criterion breakdown."""
    cls = _GRADERS.get(task_name)
    if cls is None:
        raise ValueError(f"Unknown task '{task_name}'")
    grader = cls()
    total = grader.grade(episode_data)
    breakdown = grader.grade_breakdown(episode_data)
    return {"total_score": total, "breakdown": breakdown}
