"""
Solar + Battery + Grid Tri-Source Energy System for EcoBuild.
Models rooftop solar panels, lithium battery storage, and grid import/export.
40%+ of Indian commercial buildings above 100kWp have mandated rooftop solar.

Cost hierarchy: Solar (₹0) → Battery (amortized ~₹5) → Grid (₹3-8 TOU) → Genset (₹31-45)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnergySourceState:
    solar_generation_kw: float
    battery_soc_pct: float
    battery_charging_kw: float       # Positive = charging, negative = discharging
    grid_import_kw: float
    grid_export_kw: float
    net_metering_revenue_inr: float  # Revenue from selling to grid


class SolarPanelArray:
    """
    Rooftop solar panel model.
    Output follows a bell curve peaking at solar noon (~1 PM IST).
    Monsoon cloud cover reduces output significantly.
    """

    PEAK_HOUR = 13.0  # 1 PM IST
    CLOUD_COVER_BY_SEASON = {
        "summer":       0.10,  # Clear days, high irradiance
        "monsoon":      0.65,  # Heavy cloud cover
        "post_monsoon": 0.20,
        "winter":       0.15,
    }

    def __init__(
        self,
        rated_kw: float = 20.0,
        panel_efficiency: float = 0.18,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.rated_kw = rated_kw
        self.panel_efficiency = panel_efficiency
        self.rng = rng or np.random.RandomState(0)

    def get_generation(self, hour: int, day_of_year: int, season: str = "summer") -> float:
        """
        Returns solar generation in kW for this hour.
        Output is 0 before sunrise (~6 AM) and after sunset (~7 PM).
        """
        if hour < 6 or hour >= 19:
            return 0.0

        # Bell curve centered at 1 PM
        hour_offset = hour - self.PEAK_HOUR
        irradiance_factor = np.exp(-0.5 * (hour_offset / 3.5) ** 2)

        # Cloud cover reduces output
        base_cloud = self.CLOUD_COVER_BY_SEASON.get(season, 0.15)
        cloud_noise = self.rng.uniform(-0.1, 0.1)
        cloud_factor = 1.0 - np.clip(base_cloud + cloud_noise, 0, 0.95)

        generation = self.rated_kw * irradiance_factor * cloud_factor
        return float(np.clip(generation, 0, self.rated_kw))


class BatteryStorage:
    """
    Lithium-ion battery storage with realistic SoC constraints and charge/discharge losses.
    Typical Indian commercial install: 50-100 kWh.
    Round-trip efficiency: ~90% (charge 95% × discharge 95%).
    """

    CHARGE_EFFICIENCY = 0.95
    DISCHARGE_EFFICIENCY = 0.95
    MIN_SOC = 0.10   # Don't fully discharge (protects battery life)
    MAX_SOC = 0.95   # Don't fully charge

    def __init__(self, capacity_kwh: float = 30.0, initial_soc_pct: float = 50.0):
        self.capacity_kwh = capacity_kwh
        self.soc_kwh = capacity_kwh * (initial_soc_pct / 100.0)

    def reset(self, initial_soc_pct: float = 50.0):
        self.soc_kwh = self.capacity_kwh * (initial_soc_pct / 100.0)

    @property
    def soc_pct(self) -> float:
        return (self.soc_kwh / self.capacity_kwh) * 100.0

    def charge(self, power_kw: float, dt_hours: float) -> float:
        """Charge battery. Returns actual kW accepted."""
        max_charge_kwh = (self.MAX_SOC - self.soc_kwh / self.capacity_kwh) * self.capacity_kwh
        actual_kwh = min(power_kw * dt_hours * self.CHARGE_EFFICIENCY, max_charge_kwh)
        self.soc_kwh = min(self.soc_kwh + actual_kwh, self.capacity_kwh * self.MAX_SOC)
        return actual_kwh / max(dt_hours, 1e-6)

    def discharge(self, power_kw: float, dt_hours: float) -> float:
        """Discharge battery. Returns actual kW supplied."""
        available_kwh = (self.soc_kwh / self.capacity_kwh - self.MIN_SOC) * self.capacity_kwh
        actual_kwh = min(power_kw * dt_hours / self.DISCHARGE_EFFICIENCY, available_kwh)
        self.soc_kwh = max(self.soc_kwh - actual_kwh, self.capacity_kwh * self.MIN_SOC)
        return actual_kwh * self.DISCHARGE_EFFICIENCY / max(dt_hours, 1e-6)

    def get_available_discharge_kw(self, dt_hours: float = 1/6) -> float:
        """Max discharge power available for this time step."""
        available_kwh = (self.soc_kwh / self.capacity_kwh - self.MIN_SOC) * self.capacity_kwh
        return available_kwh / max(dt_hours, 1e-6) * self.DISCHARGE_EFFICIENCY


class EnergySourceController:
    """
    Manages prioritized energy dispatch across Solar, Battery, Grid, and Genset.
    Tracks costs from each source and net metering revenue.

    Net metering rate: ₹2.5/kWh (typical DISCOM buy-back rate in India).
    """

    NET_METERING_RATE_INR = 2.5  # ₹/kWh

    def __init__(
        self,
        solar: Optional[SolarPanelArray] = None,
        battery: Optional[BatteryStorage] = None,
        has_solar: bool = True,
        has_battery: bool = True,
    ):
        self.solar = solar or SolarPanelArray()
        self.battery = battery or BatteryStorage()
        self.has_solar = has_solar
        self.has_battery = has_battery
        self.total_net_metering_revenue = 0.0

    def reset(self):
        self.battery.reset()
        self.total_net_metering_revenue = 0.0

    def dispatch(
        self,
        load_kw: float,
        hour: int,
        day_of_year: int,
        season: str,
        grid_available: bool,
        tariff_rate: float,
        battery_control: int,
        dt_hours: float = 1/6,
    ) -> EnergySourceState:
        """
        Dispatch energy from available sources to meet building load.

        Args:
            load_kw: Building total electrical load
            hour: Current hour
            day_of_year: For solar calculation
            season: For cloud cover
            grid_available: Whether grid is online
            tariff_rate: Current ₹/kWh grid rate
            battery_control: 0=auto, 1=force_charge, 2=force_discharge
            dt_hours: Step duration

        Returns:
            EnergySourceState with all source outputs and costs
        """
        solar_gen = self.solar.get_generation(hour, day_of_year, season) if self.has_solar else 0.0

        remaining_load = load_kw
        battery_charging_kw = 0.0
        grid_import_kw = 0.0
        grid_export_kw = 0.0
        net_metering_revenue = 0.0

        # Step 1: Use solar generation first (free)
        solar_to_load = min(solar_gen, remaining_load)
        remaining_load -= solar_to_load
        solar_surplus = solar_gen - solar_to_load

        # Step 2: Charge battery from solar surplus (or force charge from grid)
        if self.has_battery:
            if battery_control == 1 and grid_available:
                # Force charge from grid
                charge_kw = self.battery.charge(min(10.0, load_kw * 0.3), dt_hours)
                battery_charging_kw = charge_kw
                remaining_load += charge_kw
            elif solar_surplus > 0.5:
                # Auto: store surplus solar in battery
                charge_kw = self.battery.charge(solar_surplus, dt_hours)
                battery_charging_kw = charge_kw
                solar_surplus -= charge_kw

        # Step 3: Export remaining solar surplus to grid (net metering)
        if solar_surplus > 0.1 and grid_available:
            grid_export_kw = solar_surplus
            net_metering_revenue = grid_export_kw * dt_hours * self.NET_METERING_RATE_INR
            self.total_net_metering_revenue += net_metering_revenue

        # Step 4: Discharge battery if needed (force_discharge or auto)
        if self.has_battery and remaining_load > 0:
            if battery_control == 2 or not grid_available:
                discharge_kw = self.battery.discharge(remaining_load, dt_hours)
                battery_charging_kw -= discharge_kw
                remaining_load -= discharge_kw

        # Step 5: Import from grid for remaining load
        if remaining_load > 0 and grid_available:
            grid_import_kw = remaining_load
            remaining_load = 0.0

        return EnergySourceState(
            solar_generation_kw=solar_gen,
            battery_soc_pct=self.battery.soc_pct,
            battery_charging_kw=battery_charging_kw,
            grid_import_kw=grid_import_kw,
            grid_export_kw=grid_export_kw,
            net_metering_revenue_inr=net_metering_revenue,
        )
