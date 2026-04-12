"""
Task Configuration Registry for EcoBuild.
Centralized config for all 4 RL tasks with full physics, India features, and grader weights.
Task name is passed at reset() time — the same env instance handles all tasks.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskConfig:
    name: str
    difficulty: str
    description: str
    duration_steps: int            # Episode length in 10-min steps
    season: str                    # Weather season
    region: str                    # City/region profile
    building_type: str             # office / residential / retail

    # Physics
    thermal_mass: float            # kWh/°C  (low=cabin, high=concrete)
    insulation: float              # °C/kW (higher=better insulated)
    comfort_temp_min: float        # Lower bound of comfort zone (°C)
    comfort_temp_max: float        # Upper bound of comfort zone (°C)

    # Infrastructure flags
    has_ac: bool
    has_solar: bool
    has_battery: bool
    has_genset: bool

    # Grid config
    power_cut_hours_per_day: float
    grid_city_profile: str         # "tier1", "tier2", "delhi", etc.
    grid_mode: str                 # "scheduled" or "stochastic"

    # Grader weights (must sum to 1.0)
    grader_weights: dict

    # Documented baseline performance (filled after baseline runs)
    baseline_random: Optional[float] = None
    baseline_rule_based: Optional[float] = None
    target_score: Optional[float] = None


TASK_CONFIGS: dict[str, TaskConfig] = {

    "basic_thermostat": TaskConfig(
        name="basic_thermostat",
        difficulty="easy",
        description=(
            "Maintain comfort (20-22°C) in a well-insulated Delhi office during winter. "
            "No power cuts, no AC needed. Binary heater + lights. "
            "Thermal mass creates 30-min lag — pre-heating before occupancy is rewarded."
        ),
        duration_steps=144,          # 24 hours
        season="winter",
        region="delhi",
        building_type="office",
        thermal_mass=2.0,
        insulation=0.5,
        comfort_temp_min=20.0,
        comfort_temp_max=22.0,
        has_ac=False,
        has_solar=False,
        has_battery=False,
        has_genset=False,
        power_cut_hours_per_day=0.0,
        grid_city_profile="delhi",
        grid_mode="scheduled",
        grader_weights={
            "comfort": 0.40,
            "energy_efficiency": 0.40,
            "vacancy_penalty": 0.20,
        },
        baseline_random=0.12,
        baseline_rule_based=0.68,
        target_score=0.70,
    ),

    "day_night_tou": TaskConfig(
        name="day_night_tou",
        difficulty="medium",
        description=(
            "Delhi summer office with time-of-use electricity pricing. "
            "Peak tariff 6-10 PM (₹8/kWh) vs off-peak 10 PM-6 AM (₹3/kWh). "
            "AC required for cooling. Pre-cool during cheap morning hours to avoid peak cost."
        ),
        duration_steps=144,          # 24 hours
        season="summer",
        region="delhi",
        building_type="office",
        thermal_mass=1.5,
        insulation=0.4,
        comfort_temp_min=22.0,
        comfort_temp_max=25.0,
        has_ac=True,
        has_solar=False,
        has_battery=False,
        has_genset=False,
        power_cut_hours_per_day=0.0,
        grid_city_profile="delhi",
        grid_mode="scheduled",
        grader_weights={
            "energy_cost_inr": 0.30,
            "comfort": 0.30,
            "tou_awareness": 0.25,
            "vacancy_penalty": 0.15,
        },
        baseline_random=0.08,
        baseline_rule_based=0.45,
        target_score=0.65,
    ),

    "load_shedding_optimizer": TaskConfig(
        name="load_shedding_optimizer",
        difficulty="medium_hard",
        description=(
            "Mumbai monsoon, tier-2 city grid with 5 hrs/day scheduled power cuts. "
            "Diesel genset available but expensive (₹35/kWh). "
            "Pre-condition the building before cuts and selectively use genset for critical loads. "
            "High humidity requires AC dehumidification even during outages."
        ),
        duration_steps=288,          # 48 hours
        season="monsoon",
        region="mumbai",
        building_type="office",
        thermal_mass=2.5,
        insulation=0.45,
        comfort_temp_min=22.0,
        comfort_temp_max=26.0,
        has_ac=True,
        has_solar=False,
        has_battery=False,
        has_genset=True,
        power_cut_hours_per_day=5.0,
        grid_city_profile="tier2",
        grid_mode="scheduled",
        grader_weights={
            "comfort_during_cuts": 0.40,
            "genset_cost": 0.30,
            "pre_conditioning": 0.30,
        },
        baseline_random=0.05,
        baseline_rule_based=0.30,
        target_score=0.55,
    ),

    "multiday_optimization": TaskConfig(
        name="multiday_optimization",
        difficulty="hard",
        description=(
            "7-day Delhi optimization spanning summer-to-monsoon transition. "
            "Rooftop solar (20 kW) + battery (30 kWh) + TOU tariffs + demand charges. "
            "Festival day on day 4 (2× occupancy). AQI smog event on day 6 (outdoor AQI=380). "
            "Minimize total ₹ cost while maintaining comfort and minimizing CO₂ emissions."
        ),
        duration_steps=1008,         # 7 days × 144 steps/day
        season="summer",             # Starts summer, weather model transitions
        region="delhi",
        building_type="office",
        thermal_mass=3.0,
        insulation=0.6,
        comfort_temp_min=21.0,
        comfort_temp_max=25.0,
        has_ac=True,
        has_solar=True,
        has_battery=True,
        has_genset=False,
        power_cut_hours_per_day=1.0,
        grid_city_profile="delhi",
        grid_mode="stochastic",
        grader_weights={
            "total_cost_inr": 0.30,
            "comfort": 0.30,
            "co2_emissions": 0.20,
            "anticipation": 0.10,
            "constraint_free": 0.10,
        },
        baseline_random=0.04,
        baseline_rule_based=0.28,
        target_score=0.50,
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    """Retrieve task config by name. Raises ValueError for unknown tasks."""
    if task_name not in TASK_CONFIGS:
        available = list(TASK_CONFIGS.keys())
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: {available}"
        )
    return TASK_CONFIGS[task_name]


def list_tasks() -> list:
    """Returns list of all available task names with difficulty."""
    return [
        {"name": k, "difficulty": v.difficulty, "description": v.description}
        for k, v in TASK_CONFIGS.items()
    ]
