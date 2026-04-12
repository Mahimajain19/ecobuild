"""
Pydantic data models for EcoBuild.
Defines all structured inputs/outputs for the environment API.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
import uuid


# ─────────────────────────────────────────────
# OBSERVATION
# ─────────────────────────────────────────────

class BuildingObservation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Thermal
    indoor_temperature: float   # Current indoor temp (°C)
    outdoor_temperature: float  # Current outdoor temp (°C)
    humidity: float             # Indoor relative humidity (%)
    outdoor_humidity: float     # Outdoor relative humidity (%)

    # Occupancy
    occupancy_count: int        # Number of people currently in building
    predicted_occupancy_2h: int # Predicted occupancy 2 hours ahead

    # Time
    hour_of_day: int            # 0-23
    day_of_week: int            # 0=Monday, 6=Sunday
    time_step: int              # Current step in episode

    # Actuator states
    heater_status: int          # 0=off, 1=on
    ac_status: int              # 0=off, 1=on
    lights_status: int          # 0=off, 1=on
    fan_speed: int              # 0=off, 1=low, 2=high
    fresh_air_damper: int       # 0=closed, 1=low, 2=medium, 3=full

    # Energy
    cumulative_energy_kwh: float          # Total kWh used this episode
    electricity_price_normalized: float   # Current tariff, 0-1 scale

    # Grid
    grid_available: int                   # 1=power on, 0=cut
    grid_voltage: float                   # V (0 when cut)
    predicted_next_cut_minutes: int       # Minutes until next outage

    # Solar & Battery (0 if not installed)
    solar_generation_kw: float
    battery_soc_pct: float

    # Genset
    genset_available: int                 # 1=can start, 0=no fuel/fault
    genset_fuel_pct: float                # 0-100%

    # Air Quality
    outdoor_aqi: float                    # 0-500 AQI scale
    indoor_aqi: float                     # 0-500
    indoor_co2_ppm: float                 # ppm

    # Festival context
    days_until_next_festival: int
    festival_occupancy_mult: float        # 1.0 = normal day


# ─────────────────────────────────────────────
# ACTION
# ─────────────────────────────────────────────

class BuildingAction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    heater_control: int = Field(default=0, ge=0, le=1)   # 0=off, 1=on
    ac_control: int = Field(default=0, ge=0, le=1)        # 0=off, 1=on (AC/cooling)
    lights_control: int = Field(default=0, ge=0, le=1)    # 0=off, 1=on
    fan_speed: int = Field(default=0, ge=0, le=2)         # 0=off, 1=low, 2=high
    fresh_air_damper: int = Field(default=0, ge=0, le=3)  # 0=closed..3=full
    genset_control: int = Field(default=0, ge=0, le=1)    # 0=off, 1=on (during cuts only)
    battery_charge_rate: int = Field(default=0, ge=0, le=2)  # 0=auto, 1=force_charge, 2=force_discharge


# ─────────────────────────────────────────────
# REWARD
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    total: float
    energy_cost_inr: float          # ₹ cost this step
    comfort_penalty: float          # Penalty for temp out of range
    humidity_penalty: float         # Penalty for RH out of 40-60%
    aqi_penalty: float              # Penalty for poor indoor air quality
    vacancy_waste: float            # Penalty for running unused equipment
    constraint_violation: float     # Large penalty for extreme temps
    anticipation_bonus: float       # Bonus for pre-heating before occupancy
    net_metering_revenue: float     # Revenue from solar export (₹)
    equipment_damage_penalty: float # Penalty for running AC at unsafe voltage


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class EcoBuildState(BaseModel):
    """Episode-level server-side state. Returned by GET /state."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    episode_id: str
    task_name: str
    step_count: int
    max_steps: int
    total_energy_kwh: float
    total_cost_inr: float
    comfort_violations: int
    is_done: bool
    episode_seed: Optional[int] = None
    current_score: float = 0.0         # Rolling grader estimate
    season: str = "summer"
    region: str = "generic"
