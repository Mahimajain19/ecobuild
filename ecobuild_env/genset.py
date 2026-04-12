"""
Diesel Genset (Generator Set) Model for EcoBuild.
Models backup power generation during grid outages — a reality in 60%+ of
Indian tier-2/3 cities. Tracks fuel consumption, runtime, and maintenance state.

Real cost: ₹31-45/kWh from diesel vs ₹3-8/kWh from grid.
Agent learns to use genset selectively (critical loads only) during outages.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GensetState:
    is_running: bool
    fuel_level_liters: float
    runtime_hours: float
    maintenance_overdue: bool
    current_output_kw: float
    cost_per_kwh: float


class DieselGenset:
    """
    Diesel generator set with realistic fuel economics and maintenance model.

    Typical Indian commercial building: 62.5 kVA (50 kW rated) genset.
    Fuel consumption: ~0.35 L/kWh at full load (worse at part load).
    """

    def __init__(
        self,
        rated_kw: float = 25.0,
        tank_capacity_liters: float = 200.0,
        diesel_price_per_liter: float = 90.0,
        maintenance_interval_hours: float = 500.0,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.rated_kw = rated_kw
        self.tank_capacity = tank_capacity_liters
        self.diesel_price = diesel_price_per_liter
        self.maintenance_interval = maintenance_interval_hours
        self.rng = rng or np.random.RandomState(0)

        # State variables
        self.fuel_level = tank_capacity_liters * 0.8  # Start at 80% full
        self.runtime_hours = 0.0
        self.hours_since_maintenance = self.rng.uniform(0, 300)
        self.is_running = False
        self.current_output_kw = 0.0
        self.total_fuel_used = 0.0
        self.total_cost = 0.0

    def reset(self, rng: Optional[np.random.RandomState] = None):
        if rng is not None:
            self.rng = rng
        self.fuel_level = self.tank_capacity * self.rng.uniform(0.6, 0.95)
        self.runtime_hours = 0.0
        self.hours_since_maintenance = self.rng.uniform(0, 400)
        self.is_running = False
        self.current_output_kw = 0.0
        self.total_fuel_used = 0.0
        self.total_cost = 0.0

    @property
    def fuel_pct(self) -> float:
        return self.fuel_level / self.tank_capacity * 100.0

    @property
    def maintenance_overdue(self) -> bool:
        return self.hours_since_maintenance >= self.maintenance_interval

    def can_start(self) -> bool:
        """Returns True if genset can be started (fuel OK, not critically overdue)."""
        return self.fuel_level > self.tank_capacity * 0.05

    def _fuel_consumption_rate(self, load_kw: float) -> float:
        """
        Returns litres/kWh at given load.
        Part-load is less efficient than full load (diesel engine characteristic).
        """
        load_fraction = load_kw / max(self.rated_kw, 1.0)
        if load_fraction >= 0.75:
            return 0.35  # Efficient zone: L/kWh
        elif load_fraction >= 0.50:
            return 0.40
        elif load_fraction >= 0.25:
            return 0.50
        else:
            return 0.65  # Very inefficient at low load

    def get_cost_per_kwh(self, load_kw: float) -> float:
        """Returns ₹/kWh for generating at this load."""
        fuel_rate = self._fuel_consumption_rate(load_kw)
        base_cost = fuel_rate * self.diesel_price

        # Maintenance penalty: overdue genset is 20% less efficient
        if self.maintenance_overdue:
            base_cost *= 1.20

        return float(base_cost)

    def step(
        self,
        control: int,
        building_load_kw: float,
        dt_hours: float = 1 / 6,
    ) -> dict:
        """
        Update genset state for one time step.

        Args:
            control: 0=off, 1=on (agent decision)
            building_load_kw: Load the genset must supply
            dt_hours: Step duration in hours

        Returns:
            dict with output_kw, cost_inr, fuel_used_liters, can_supply_full_load
        """
        result = {
            "output_kw": 0.0,
            "cost_inr": 0.0,
            "fuel_used_liters": 0.0,
            "cost_per_kwh": 0.0,
            "can_supply_full_load": False,
        }

        if control == 1 and self.can_start():
            self.is_running = True
            # Clamp to rated capacity
            actual_output = min(building_load_kw, self.rated_kw)
            self.current_output_kw = actual_output

            fuel_rate = self._fuel_consumption_rate(actual_output)
            fuel_used = fuel_rate * actual_output * dt_hours
            energy_kwh = actual_output * dt_hours
            cost = energy_kwh * self.get_cost_per_kwh(actual_output)

            self.fuel_level = max(0.0, self.fuel_level - fuel_used)
            self.runtime_hours += dt_hours
            self.hours_since_maintenance += dt_hours
            self.total_fuel_used += fuel_used
            self.total_cost += cost

            result["output_kw"] = actual_output
            result["cost_inr"] = cost
            result["fuel_used_liters"] = fuel_used
            result["cost_per_kwh"] = self.get_cost_per_kwh(actual_output)
            result["can_supply_full_load"] = actual_output >= building_load_kw
        else:
            self.is_running = False
            self.current_output_kw = 0.0

        return result

    def get_state(self) -> GensetState:
        return GensetState(
            is_running=self.is_running,
            fuel_level_liters=self.fuel_level,
            runtime_hours=self.runtime_hours,
            maintenance_overdue=self.maintenance_overdue,
            current_output_kw=self.current_output_kw,
            cost_per_kwh=self.get_cost_per_kwh(self.current_output_kw),
        )
