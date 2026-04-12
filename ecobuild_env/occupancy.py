"""
Occupancy model for EcoBuild.
Returns occupancy COUNT (not binary 0/1) representing number of people in the building.
Uses seeded RNG for reproducibility.
"""

import numpy as np
from typing import Optional


# Building capacity profiles
BUILDING_CAPACITIES = {
    "office":       {"max_occupancy": 30, "core_hours": (9, 18)},
    "residential":  {"max_occupancy": 8,  "core_hours": (18, 23)},
    "retail":       {"max_occupancy": 50, "core_hours": (10, 21)},
}


def is_occupied_fixed(
    hour: int,
    day_of_week: int,
    building_type: str = "office",
    festival_mult: float = 1.0,
) -> int:
    """
    Fixed occupancy schedule returning person COUNT.
    Weekday office: arrives 8-10 AM, leaves 5-7 PM, lunch dip at noon.

    Args:
        hour: Hour of day (0-23)
        day_of_week: 0=Monday, 6=Sunday
        building_type: "office", "residential", "retail"
        festival_mult: Multiplier from festival calendar

    Returns:
        Integer count of occupants
    """
    max_occ = BUILDING_CAPACITIES.get(building_type, BUILDING_CAPACITIES["office"])["max_occupancy"]
    core_start, core_end = BUILDING_CAPACITIES.get(building_type, BUILDING_CAPACITIES["office"])["core_hours"]

    if building_type == "office":
        if day_of_week >= 5:  # Weekend
            return 0

        if hour < 8 or hour >= 19:
            return 0
        elif 8 <= hour < 9:
            count = int(max_occ * 0.25)   # Early arrivals
        elif 9 <= hour < 12:
            count = int(max_occ * 0.90)   # Core hours
        elif 12 <= hour < 14:
            count = int(max_occ * 0.60)   # Lunch dip
        elif 14 <= hour < 17:
            count = int(max_occ * 0.95)   # Post-lunch peak
        elif 17 <= hour < 18:
            count = int(max_occ * 0.60)   # People leaving
        else:
            count = int(max_occ * 0.15)   # Stragglers

        return int(min(max_occ * 2, count * festival_mult))

    elif building_type == "residential":
        if 7 <= hour < 22:
            count = int(max_occ * (0.4 + 0.5 * (1 if hour >= 18 else 0)))
        else:
            count = max_occ  # Night: full household sleeping
        return int(min(max_occ * 2, count * festival_mult))

    return 0


def is_occupied_stochastic(
    hour: int,
    day_of_week: int,
    building_type: str = "office",
    festival_mult: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> int:
    """
    Stochastic occupancy using Poisson arrival model.
    Returns integer count of occupants.

    Args:
        hour: Hour of day
        day_of_week: 0=Monday, 6=Sunday
        building_type: Building type
        festival_mult: Festival occupancy multiplier
        rng: Seeded RandomState

    Returns:
        Integer occupant count
    """
    if rng is None:
        rng = np.random.RandomState()

    # Get the expected (mean) count from fixed schedule
    mean_count = is_occupied_fixed(hour, day_of_week, building_type, festival_mult)

    if mean_count == 0:
        # Small probability of stragglers/guards
        return int(rng.poisson(0.3))

    # Poisson noise around expected count
    return int(np.clip(rng.poisson(mean_count), 0, mean_count * 3))


def get_predicted_occupancy(
    hour: int,
    day_of_week: int,
    lookahead_hours: int = 2,
    building_type: str = "office",
) -> int:
    """
    Returns predicted occupancy count N hours ahead (deterministic).
    Used as a lookahead observation feature for anticipatory control.
    """
    future_hour = (hour + lookahead_hours) % 24
    return is_occupied_fixed(future_hour, day_of_week, building_type)
