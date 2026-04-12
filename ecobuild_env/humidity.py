"""
Humidity dynamics for EcoBuild.
Models indoor/outdoor relative humidity transitions including AC dehumidification,
occupant moisture generation, and ventilation effects.
Critical for Indian monsoon season modeling (Jul-Sep: 70-95% RH outdoors).
"""

import numpy as np
from typing import Optional


# Outdoor humidity profiles by city and season (% RH ranges)
HUMIDITY_PROFILES = {
    "delhi": {
        "summer":       {"base": 30, "amplitude": 15},
        "monsoon":      {"base": 80, "amplitude": 12},
        "post_monsoon": {"base": 55, "amplitude": 15},
        "winter":       {"base": 70, "amplitude": 18},
    },
    "mumbai": {
        "summer":       {"base": 65, "amplitude": 10},
        "monsoon":      {"base": 88, "amplitude": 6},
        "post_monsoon": {"base": 72, "amplitude": 10},
        "winter":       {"base": 65, "amplitude": 10},
    },
    "bangalore": {
        "summer":       {"base": 45, "amplitude": 15},
        "monsoon":      {"base": 75, "amplitude": 10},
        "post_monsoon": {"base": 70, "amplitude": 12},
        "winter":       {"base": 60, "amplitude": 12},
    },
    "generic": {
        "summer":       {"base": 40, "amplitude": 15},
        "monsoon":      {"base": 80, "amplitude": 12},
        "post_monsoon": {"base": 60, "amplitude": 15},
        "winter":       {"base": 65, "amplitude": 15},
    },
}

# Comfort range
COMFORT_RH_MIN = 40.0
COMFORT_RH_MAX = 60.0


def get_outdoor_humidity(
    hour: int,
    day_of_year: int,
    region: str = "generic",
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """
    Returns outdoor relative humidity (%) for a given hour and day.
    Humidity is highest in early morning (pre-dawn) and lowest in early afternoon.

    Args:
        hour: Hour of day (0-23)
        day_of_year: Day of year (0-364)
        region: City/region name
        rng: Seeded random state for reproducibility
    """
    season = _get_season(day_of_year)
    profile = HUMIDITY_PROFILES.get(region, HUMIDITY_PROFILES["generic"])[season]

    base = profile["base"]
    amp = profile["amplitude"]

    # Peak humidity around 5-6 AM, lowest around 2-3 PM
    daily_cycle = amp * np.cos(2 * np.pi * (hour - 5) / 24)

    # Small random noise
    noise = 0.0
    if rng is not None:
        noise = rng.uniform(-3.0, 3.0)

    return float(np.clip(base + daily_cycle + noise, 10.0, 100.0))


def update_indoor_humidity(
    indoor_rh: float,
    outdoor_rh: float,
    ac_on: int,
    fan_speed: int,
    occupancy_count: int,
    dt_hours: float = 1 / 6,
) -> float:
    """
    Updates indoor relative humidity for one time step (default: 10 min = 1/6 hr).

    Physics:
    - AC dehumidifies at ~0.5 L/hr per unit → lowers RH
    - Each person emits ~50 g/hr moisture → raises RH
    - Ventilation fan mixes indoor/outdoor air → moves toward outdoor RH

    Args:
        indoor_rh: Current indoor RH (%)
        outdoor_rh: Current outdoor RH (%)
        ac_on: 1 if AC is running
        fan_speed: 0=off, 1=low, 2=high
        occupancy_count: Number of people in building
        dt_hours: Time step duration in hours

    Returns:
        Next indoor RH (%)
    """
    # AC dehumidification effect: -3% RH per hour when running
    ac_effect = -3.0 * ac_on * dt_hours

    # Occupant moisture: each person raises RH by ~0.2%/hr in a typical room
    occupant_effect = 0.2 * occupancy_count * dt_hours

    # Ventilation: pull indoor towards outdoor at rate dependent on fan speed
    vent_rate = {0: 0.05, 1: 0.15, 2: 0.30}.get(fan_speed, 0.05)
    vent_effect = vent_rate * (outdoor_rh - indoor_rh) * dt_hours

    next_rh = indoor_rh + ac_effect + occupant_effect + vent_effect
    return float(np.clip(next_rh, 10.0, 100.0))


def humidity_comfort_penalty(indoor_rh: float) -> float:
    """
    Returns a comfort penalty (0.0 = comfortable, >0 = uncomfortable)
    based on indoor RH. Comfort zone: 40-60%.
    """
    if COMFORT_RH_MIN <= indoor_rh <= COMFORT_RH_MAX:
        return 0.0
    elif indoor_rh < COMFORT_RH_MIN:
        return (COMFORT_RH_MIN - indoor_rh) * 0.05  # too dry
    else:
        return (indoor_rh - COMFORT_RH_MAX) * 0.08  # too humid (worse)


def _get_season(day_of_year: int) -> str:
    """Map day of year to Indian season."""
    if 60 <= day_of_year < 180:
        return "summer"
    elif 180 <= day_of_year < 270:
        return "monsoon"
    elif 270 <= day_of_year < 335:
        return "post_monsoon"
    else:
        return "winter"
