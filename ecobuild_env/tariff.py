"""
Indian Electricity Tariff Model
Simulates Time-of-Use (TOU) pricing, demand charges, and power factor penalties
based on real Indian state electricity board tariff structures.
"""

from dataclasses import dataclass, field
from typing import Optional


# Regional tariff profiles (approx. real values in ₹/kWh)
REGIONAL_TARIFFS = {
    "delhi": {
        "off_peak": 3.00,   # 22:00 - 06:00
        "normal": 5.50,     # 06:00 - 18:00
        "peak": 8.00,       # 18:00 - 22:00
        "demand_charge_per_kva": 300.0,
        "fixed_charge_per_month": 125.0,
    },
    "mumbai": {
        "off_peak": 2.80,
        "normal": 5.20,
        "peak": 9.10,
        "demand_charge_per_kva": 280.0,
        "fixed_charge_per_month": 100.0,
    },
    "bangalore": {
        "off_peak": 3.50,
        "normal": 6.00,
        "peak": 7.50,
        "demand_charge_per_kva": 250.0,
        "fixed_charge_per_month": 90.0,
    },
    "chennai": {
        "off_peak": 3.20,
        "normal": 5.80,
        "peak": 8.50,
        "demand_charge_per_kva": 260.0,
        "fixed_charge_per_month": 110.0,
    },
    "generic": {
        "off_peak": 3.00,
        "normal": 5.50,
        "peak": 8.00,
        "demand_charge_per_kva": 300.0,
        "fixed_charge_per_month": 100.0,
    },
}


class IndianElectricityTariff:
    """
    Simulates Indian electricity tariff structure including:
    - Time-of-Use (TOU) pricing with peak/normal/off-peak slots
    - Demand charges based on peak kW demand
    - Power factor penalty for simultaneous heavy loads
    """

    def __init__(self, region: str = "generic"):
        region = region.lower()
        if region not in REGIONAL_TARIFFS:
            region = "generic"
        profile = REGIONAL_TARIFFS[region]
        self.region = region
        self.off_peak_rate = profile["off_peak"]
        self.normal_rate = profile["normal"]
        self.peak_rate = profile["peak"]
        self.demand_charge_per_kva = profile["demand_charge_per_kva"]
        self.fixed_charge = profile["fixed_charge_per_month"]

        # Episode-level tracking
        self.peak_demand_kw: float = 0.0
        self.total_cost_inr: float = 0.0

    def reset(self):
        """Reset episode trackers."""
        self.peak_demand_kw = 0.0
        self.total_cost_inr = 0.0

    def get_tariff_slot(self, hour: int) -> str:
        """Returns the tariff slot name for a given hour."""
        if 22 <= hour or hour < 6:
            return "off_peak"
        elif 18 <= hour < 22:
            return "peak"
        else:
            return "normal"

    def get_current_tariff(self, hour: int) -> float:
        """Returns ₹/kWh for the given hour."""
        slot = self.get_tariff_slot(hour)
        return {
            "off_peak": self.off_peak_rate,
            "normal": self.normal_rate,
            "peak": self.peak_rate,
        }[slot]

    def get_normalized_tariff(self, hour: int) -> float:
        """Returns tariff normalized to [0, 1] for use as an RL observation feature."""
        rate = self.get_current_tariff(hour)
        return (rate - self.off_peak_rate) / max(1e-6, self.peak_rate - self.off_peak_rate)

    def get_step_cost(self, energy_kwh: float, load_kw: float, hour: int) -> float:
        """
        Calculate ₹ cost for this step.

        Args:
            energy_kwh: Energy consumed this step (kWh)
            load_kw: Instantaneous load (kW) for demand tracking
            hour: Current hour of day

        Returns:
            Cost in ₹ for this step (energy cost only; demand charge is episode-level)
        """
        # Track peak demand for demand charge calculation
        if load_kw > self.peak_demand_kw:
            self.peak_demand_kw = load_kw

        # Power factor penalty: running all loads simultaneously costs more
        # Simplified: if load > 80% of rated, apply 10% surcharge
        pf_penalty = 1.10 if load_kw > 8.0 else 1.0

        rate = self.get_current_tariff(hour)
        cost = energy_kwh * rate * pf_penalty
        self.total_cost_inr += cost
        return float(cost)

    def get_episode_demand_charge(self) -> float:
        """Returns the demand charge (₹) based on peak kW demand this episode."""
        # Demand charge is monthly but we prorate by episode duration
        # For a 24h episode it's 1/30th of monthly demand charge
        return self.peak_demand_kw * self.demand_charge_per_kva / 30.0

    def get_total_episode_cost(self) -> float:
        """Total cost including energy + demand charges."""
        return self.total_cost_inr + self.get_episode_demand_charge()
