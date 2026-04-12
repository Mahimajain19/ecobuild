"""
Air Quality (AQI) Model for EcoBuild.
Models PM2.5-based AQI for Indian cities, indoor CO2 buildup from occupants,
and the trade-off between fresh air ventilation and energy/filtration cost.

Unique to India: Delhi winter AQI routinely hits 400-500.
Opening fresh air dampers during smog events pumps polluted air inside.
The agent must balance CO2 (needs fresh air) vs PM2.5 (avoid fresh air).
"""

import numpy as np
from typing import Optional


# Seasonal AQI profiles by city (PM2.5 equivalent AQI)
AQI_PROFILES = {
    "delhi": {
        "summer":       {"base": 120, "amplitude": 40},
        "monsoon":      {"base": 60,  "amplitude": 25},   # Rain washes particles
        "post_monsoon": {"base": 200, "amplitude": 80},   # Stubble burning
        "winter":       {"base": 350, "amplitude": 120},  # Infamous Delhi smog
    },
    "mumbai": {
        "summer":       {"base": 80,  "amplitude": 30},
        "monsoon":      {"base": 50,  "amplitude": 20},
        "post_monsoon": {"base": 110, "amplitude": 40},
        "winter":       {"base": 130, "amplitude": 50},
    },
    "bangalore": {
        "summer":       {"base": 60,  "amplitude": 25},
        "monsoon":      {"base": 40,  "amplitude": 15},
        "post_monsoon": {"base": 80,  "amplitude": 30},
        "winter":       {"base": 90,  "amplitude": 35},
    },
    "generic": {
        "summer":       {"base": 100, "amplitude": 35},
        "monsoon":      {"base": 55,  "amplitude": 20},
        "post_monsoon": {"base": 150, "amplitude": 60},
        "winter":       {"base": 200, "amplitude": 80},
    },
}

# AQI guidelines
AQI_GOOD = 50
AQI_MODERATE = 100
AQI_UNHEALTHY_SENSITIVE = 150
AQI_UNHEALTHY = 200
AQI_VERY_UNHEALTHY = 300
AQI_HAZARDOUS = 400

# CO2 thresholds (ppm)
CO2_FRESH = 400
CO2_ACCEPTABLE = 1000
CO2_STUFFY = 1500
CO2_HEALTH_RISK = 2000

# HEPA filtration efficiency (removes PM2.5)
HEPA_EFFICIENCY = 0.95


def get_outdoor_aqi(
    hour: int,
    day_of_year: int,
    region: str = "generic",
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """
    Returns outdoor AQI for a given hour and day.
    AQI is worst in early morning (5-8 AM, temperature inversion) and
    improves midday as boundary layer rises and wind picks up.
    """
    season = _get_season(day_of_year)
    profile = AQI_PROFILES.get(region, AQI_PROFILES["generic"])[season]

    base = profile["base"]
    amp = profile["amplitude"]

    # Worst at 6 AM (temperature inversion), best at 2-3 PM
    daily_cycle = amp * np.cos(2 * np.pi * (hour - 6) / 24)

    noise = 0.0
    if rng is not None:
        noise = rng.uniform(-20.0, 20.0)

    return float(np.clip(base + daily_cycle + noise, 0, 500))


def update_indoor_aqi(
    indoor_aqi: float,
    outdoor_aqi: float,
    fresh_air_damper: int,
    filtration_on: bool = True,
    dt_hours: float = 1 / 6,
) -> float:
    """
    Updates indoor AQI for one time step.

    - Fresh air damper OPEN: indoor moves toward outdoor AQI (infiltration)
    - HEPA filtration: removes 95% of PM2.5 from circulating air
    - Natural infiltration (leakage) always occurs at ~5%/hr rate

    Args:
        indoor_aqi: Current indoor AQI
        outdoor_aqi: Current outdoor AQI
        fresh_air_damper: 0=closed, 1=low(15%), 2=medium(50%), 3=full(100%)
        filtration_on: Whether HEPA/filtration is running
        dt_hours: Time step in hours
    """
    damper_openness = {0: 0.02, 1: 0.15, 2: 0.50, 3: 1.0}.get(fresh_air_damper, 0.02)

    # Infiltration: damper + natural leakage
    infiltration_rate = damper_openness + 0.02
    fresh_air_effect = infiltration_rate * (outdoor_aqi - indoor_aqi) * dt_hours

    # Filtration: reduces indoor AQI by efficiency × current level
    filtration_effect = 0.0
    if filtration_on:
        filtration_effect = -HEPA_EFFICIENCY * indoor_aqi * 0.3 * dt_hours

    next_aqi = indoor_aqi + fresh_air_effect + filtration_effect
    return float(np.clip(next_aqi, 0, 500))


def update_indoor_co2(
    indoor_co2: float,
    occupancy_count: int,
    fresh_air_damper: int,
    dt_hours: float = 1 / 6,
) -> float:
    """
    Updates indoor CO2 concentration (ppm).

    - Each person generates ~0.3 L/min CO2 = ~18000 ppm·L/hr (in typical room volume)
    - Simplified: each person raises CO2 by ~15 ppm/hr in a 200m² office
    - Fresh air dilutes CO2 toward outdoor baseline (~420 ppm)

    Args:
        indoor_co2: Current indoor CO2 (ppm)
        occupancy_count: Number of people
        fresh_air_damper: 0=closed, 1=low, 2=medium, 3=full
        dt_hours: Time step
    """
    OUTDOOR_CO2 = 420.0
    damper_openness = {0: 0.02, 1: 0.15, 2: 0.50, 3: 1.0}.get(fresh_air_damper, 0.02)

    person_effect = occupancy_count * 15.0 * dt_hours
    ventilation_effect = damper_openness * (OUTDOOR_CO2 - indoor_co2) * dt_hours

    next_co2 = indoor_co2 + person_effect + ventilation_effect
    return float(np.clip(next_co2, 400, 5000))


def aqi_comfort_penalty(indoor_aqi: float, indoor_co2: float) -> float:
    """
    Returns penalty for poor air quality (AQI + CO2).
    Both being bad at the same time is worse than each alone.
    """
    aqi_penalty = 0.0
    if indoor_aqi > AQI_UNHEALTHY:
        aqi_penalty = (indoor_aqi - AQI_UNHEALTHY) * 0.002
    elif indoor_aqi > AQI_MODERATE:
        aqi_penalty = (indoor_aqi - AQI_MODERATE) * 0.001

    co2_penalty = 0.0
    if indoor_co2 > CO2_HEALTH_RISK:
        co2_penalty = (indoor_co2 - CO2_HEALTH_RISK) * 0.001
    elif indoor_co2 > CO2_ACCEPTABLE:
        co2_penalty = (indoor_co2 - CO2_ACCEPTABLE) * 0.0005

    return float(aqi_penalty + co2_penalty)


def _get_season(day_of_year: int) -> str:
    if 60 <= day_of_year < 180:
        return "summer"
    elif 180 <= day_of_year < 270:
        return "monsoon"
    elif 270 <= day_of_year < 335:
        return "post_monsoon"
    else:
        return "winter"
