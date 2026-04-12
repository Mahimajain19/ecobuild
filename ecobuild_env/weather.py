"""
Regional Seasonal Weather Model for India — EcoBuild.
Replaces the simple sinusoidal function with a city-aware, season-aware model.

Indian seasons are extreme and asymmetric:
- Mumbai July: 35°C + 90% humidity + daily rain
- Delhi January: 4°C + 85% humidity (dense fog)
- Bangalore year-round: mild 20-28°C (pleasant)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class WeatherConditions:
    temperature: float        # °C
    humidity: float           # % RH
    rainfall_intensity: int   # 0=none, 1=light, 2=moderate, 3=heavy
    solar_irradiance: float   # 0-1 normalized (cloud factor)
    season: str


# Regional weather profiles per season
WEATHER_PROFILES = {
    "delhi": {
        "summer":       {"base_temp": 38, "amplitude": 8,  "humidity_base": 30, "rain_prob": 0.02},
        "monsoon":      {"base_temp": 33, "amplitude": 5,  "humidity_base": 80, "rain_prob": 0.55},
        "post_monsoon": {"base_temp": 26, "amplitude": 7,  "humidity_base": 55, "rain_prob": 0.05},
        "winter":       {"base_temp": 10, "amplitude": 8,  "humidity_base": 75, "rain_prob": 0.08},
    },
    "mumbai": {
        "summer":       {"base_temp": 33, "amplitude": 4,  "humidity_base": 65, "rain_prob": 0.05},
        "monsoon":      {"base_temp": 30, "amplitude": 3,  "humidity_base": 90, "rain_prob": 0.70},
        "post_monsoon": {"base_temp": 30, "amplitude": 4,  "humidity_base": 72, "rain_prob": 0.08},
        "winter":       {"base_temp": 25, "amplitude": 5,  "humidity_base": 65, "rain_prob": 0.02},
    },
    "bangalore": {
        "summer":       {"base_temp": 28, "amplitude": 6,  "humidity_base": 45, "rain_prob": 0.10},
        "monsoon":      {"base_temp": 24, "amplitude": 4,  "humidity_base": 80, "rain_prob": 0.45},
        "post_monsoon": {"base_temp": 24, "amplitude": 5,  "humidity_base": 70, "rain_prob": 0.20},
        "winter":       {"base_temp": 20, "amplitude": 6,  "humidity_base": 55, "rain_prob": 0.05},
    },
    "chennai": {
        "summer":       {"base_temp": 35, "amplitude": 5,  "humidity_base": 70, "rain_prob": 0.05},
        "monsoon":      {"base_temp": 30, "amplitude": 4,  "humidity_base": 85, "rain_prob": 0.40},
        "post_monsoon": {"base_temp": 28, "amplitude": 4,  "humidity_base": 80, "rain_prob": 0.35},
        "winter":       {"base_temp": 25, "amplitude": 5,  "humidity_base": 72, "rain_prob": 0.10},
    },
    "generic": {
        "summer":       {"base_temp": 35, "amplitude": 8,  "humidity_base": 40, "rain_prob": 0.05},
        "monsoon":      {"base_temp": 30, "amplitude": 5,  "humidity_base": 82, "rain_prob": 0.55},
        "post_monsoon": {"base_temp": 25, "amplitude": 7,  "humidity_base": 58, "rain_prob": 0.10},
        "winter":       {"base_temp": 15, "amplitude": 8,  "humidity_base": 68, "rain_prob": 0.08},
    },
}


def get_season(day_of_year: int) -> str:
    """Map day of year to Indian meteorological season."""
    if 60 <= day_of_year < 180:
        return "summer"
    elif 180 <= day_of_year < 270:
        return "monsoon"
    elif 270 <= day_of_year < 335:
        return "post_monsoon"
    else:
        return "winter"


class IndianWeatherModel:
    """
    Physics-based Indian weather model providing temperature, humidity,
    rainfall, and solar irradiance per step.
    All randomness is seeded through rng for reproducibility.
    """

    def __init__(
        self,
        region: str = "generic",
        season: Optional[str] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.region = region.lower() if region.lower() in WEATHER_PROFILES else "generic"
        self.forced_season = season  # If set, overrides day-of-year derived season
        self.rng = rng or np.random.RandomState(0)

    def reset(self, rng: Optional[np.random.RandomState] = None, season: Optional[str] = None):
        if rng is not None:
            self.rng = rng
        if season is not None:
            self.forced_season = season

    def get_conditions(self, hour: int, day_of_year: int) -> WeatherConditions:
        """
        Returns weather conditions for given hour and day-of-year.
        Temperature peaks at 2 PM, minimum at 5 AM (realistic diurnal cycle).

        Args:
            hour: Hour of day (0-23)
            day_of_year: Day of year (0-364)

        Returns:
            WeatherConditions dataclass
        """
        season = self.forced_season or get_season(day_of_year)
        profile = WEATHER_PROFILES[self.region][season]

        base_temp = profile["base_temp"]
        amplitude = profile["amplitude"]

        # Daily temp cycle: min at 5 AM, max at 2 PM
        temp_cycle = amplitude * np.sin(2 * np.pi * (hour - 5) / 24)
        temp_noise = self.rng.uniform(-1.5, 1.5)
        temperature = float(np.clip(base_temp + temp_cycle + temp_noise, -5, 50))

        # Humidity (highest at dawn, lowest at peak temp)
        humidity_base = profile["humidity_base"]
        humidity_cycle = 10.0 * np.cos(2 * np.pi * (hour - 5) / 24)
        humidity_noise = self.rng.uniform(-3.0, 3.0)
        humidity = float(np.clip(humidity_base + humidity_cycle + humidity_noise, 10, 100))

        # Rainfall (probabilistic each step)
        rain_prob = profile["rain_prob"]
        # Rain more likely in afternoon/evening during monsoon
        if season == "monsoon" and 12 <= hour <= 20:
            rain_prob *= 1.5
        rain_roll = self.rng.random()
        if rain_roll < rain_prob * 0.3:
            rainfall = 3  # Heavy
        elif rain_roll < rain_prob * 0.6:
            rainfall = 2  # Moderate
        elif rain_roll < rain_prob:
            rainfall = 1  # Light
        else:
            rainfall = 0  # None

        # Solar irradiance (affected by rain/cloud cover)
        cloud_penalty = {0: 0.0, 1: 0.25, 2: 0.55, 3: 0.80}.get(rainfall, 0.0)
        if hour < 6 or hour >= 19:
            irradiance = 0.0
        else:
            hour_factor = np.exp(-0.5 * ((hour - 13) / 3.5) ** 2)
            irradiance = float(np.clip(hour_factor * (1 - cloud_penalty), 0, 1))

        return WeatherConditions(
            temperature=temperature,
            humidity=humidity,
            rainfall_intensity=rainfall,
            solar_irradiance=irradiance,
            season=season,
        )

    def get_outdoor_temp(self, hour: int, day_of_year: int) -> float:
        """Convenience method — returns just temperature."""
        return self.get_conditions(hour, day_of_year).temperature
