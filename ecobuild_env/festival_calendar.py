"""
Indian Festival Calendar for EcoBuild.
Models energy demand spikes, occupancy surges, and extended operating hours
around major Indian festivals — unique dynamics absent from Western RL environments.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FestivalProfile:
    name: str
    base_day: int           # Approximate day of year (0-364)
    occupancy_mult: float   # Multiplier on normal occupancy (1.0 = normal)
    lighting_mult: float    # Decorative lighting extra load multiplier
    hours_extended: int     # Extra hours building stays open beyond normal
    description: str


# Pre-defined festival profiles (approximate day-of-year for a generic year)
FESTIVALS = [
    FestivalProfile("Makar Sankranti",  14, 1.1, 1.2, 0,  "Kite flying, mild activity spike"),
    FestivalProfile("Holi",             70, 0.2, 1.5, 2,  "Office holiday + evening celebrations"),
    FestivalProfile("Eid ul-Fitr",     100, 0.3, 1.3, 0,  "Half-day / holiday for many"),
    FestivalProfile("Independence Day", 228, 0.5, 2.0, 0,  "Public holiday, flag ceremonies"),
    FestivalProfile("Navratri",        275, 1.3, 2.5, 4,  "Evening gatherings, extended hours"),
    FestivalProfile("Dussehra",        285, 0.4, 3.0, 2,  "Evening events, reduced daytime"),
    FestivalProfile("Diwali",          300, 2.0, 5.0, 6,  "Peak event: 2× occupancy, massive lighting"),
    FestivalProfile("Bhai Dooj",       302, 1.5, 2.0, 2,  "2 days after Diwali, elevated activity"),
    FestivalProfile("Christmas",       359, 0.5, 3.0, 4,  "Metro offices at 50% with decorations"),
    FestivalProfile("New Year Eve",    364, 0.3, 4.0, 8,  "Party/events, very late hours"),
]


class FestivalCalendar:
    """
    Provides festival-aware occupancy and load multipliers.
    All randomness is seeded for reproducibility per episode.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState(0)
        # Apply ±3 day jitter to each festival to simulate year-to-year variation
        self._jittered = [
            (f, f.base_day + self.rng.randint(-3, 4))
            for f in FESTIVALS
        ]

    def reset(self, rng: Optional[np.random.RandomState] = None):
        if rng is not None:
            self.rng = rng
        self._jittered = [
            (f, f.base_day + self.rng.randint(-3, 4))
            for f in FESTIVALS
        ]

    def get_festival_today(self, day_of_year: int) -> Optional[FestivalProfile]:
        """Returns the FestivalProfile if today is a festival day, else None."""
        for festival, jday in self._jittered:
            if jday == day_of_year:
                return festival
        return None

    def days_until_next_festival(self, day_of_year: int) -> int:
        """Returns how many days until the next festival."""
        min_days = 365
        for _, jday in self._jittered:
            delta = (jday - day_of_year) % 365
            if delta == 0:
                return 0
            if delta < min_days:
                min_days = delta
        return min_days

    def get_multipliers(self, day_of_year: int) -> dict:
        """
        Returns occupancy, lighting, and hours multipliers for the given day.
        Falls back to neutral values (1.0) on non-festival days.
        """
        festival = self.get_festival_today(day_of_year)
        if festival:
            return {
                "occupancy_mult": festival.occupancy_mult,
                "lighting_mult": festival.lighting_mult,
                "hours_extended": festival.hours_extended,
                "festival_name": festival.name,
                "is_festival": True,
            }
        return {
            "occupancy_mult": 1.0,
            "lighting_mult": 1.0,
            "hours_extended": 0,
            "festival_name": None,
            "is_festival": False,
        }
