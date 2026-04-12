"""
RC Thermal Model for EcoBuild.
Replaces simple weighted average with physics-based RC (Resistor-Capacitor) model.

dT/dt = (T_out - T_in)/(R*C) + Q_heater/C + Q_solar/C - Q_ac/C + Q_occupants/C

Where:
  R = Thermal resistance (insulation quality) — higher R = better insulated
  C = Thermal capacitance (building mass) — higher C = slower temp change
  Q = Heat flux (kW)
"""

import numpy as np


# Constants
STEPS_PER_HOUR = 6           # 10-min steps
DT_HOURS = 1 / STEPS_PER_HOUR
PERSON_HEAT_WATTS = 80.0     # Each person generates 80W of sensible heat
KW_PER_PERSON = PERSON_HEAT_WATTS / 1000.0


def update_temperature(
    current_temp: float,
    outdoor_temp: float,
    heater_on: int,
    ac_on: int = 0,
    occupancy_count: int = 0,
    solar_gain_kw: float = 0.0,
    thermal_mass: float = 2.0,
    insulation: float = 0.5,
    heater_kw: float = 5.0,
    ac_kw: float = 3.5,
    dt_hours: float = DT_HOURS,
) -> float:
    """
    Update indoor temperature using RC thermal model.

    Args:
        current_temp: Current indoor temperature (°C)
        outdoor_temp: Current outdoor temperature (°C)
        heater_on: 1 if heater is running
        ac_on: 1 if AC/cooling is running
        occupancy_count: Number of people (each adds 80W heat)
        solar_gain_kw: Solar heat gain through windows (kW)
        thermal_mass: Building thermal capacitance (kWh/°C) — higher = slower
        insulation: Building thermal resistance (°C/kW) — higher = better insulated
        heater_kw: Heater rated power (kW)
        ac_kw: AC rated cooling capacity (kW)
        dt_hours: Time step duration (hours)

    Returns:
        Next indoor temperature (°C)
    """
    # Heat exchange with outside (conduction/convection through envelope)
    q_envelope = (outdoor_temp - current_temp) / insulation  # kW (positive = heat gain from outside)

    # Internal heat sources
    q_heater = heater_kw * heater_on          # Heating (positive)
    q_ac = -ac_kw * ac_on                     # Cooling (negative)
    q_occupants = KW_PER_PERSON * occupancy_count  # People (positive)
    q_solar = solar_gain_kw                   # Solar gain (positive)

    # Net heat flux
    q_net = q_envelope + q_heater + q_ac + q_occupants + q_solar

    # Temperature change: dT = Q_net × dt / C
    delta_t = (q_net * dt_hours) / thermal_mass

    next_temp = current_temp + delta_t
    return float(round(np.clip(next_temp, -5.0, 50.0), 2))


def calculate_solar_gain(
    hour: int,
    solar_irradiance: float = 0.5,
    window_area_ratio: float = 0.25,
    building_floor_area_m2: float = 200.0,
) -> float:
    """
    Calculate solar heat gain through windows (kW).

    Args:
        hour: Hour of day (peak gain around 10AM-2PM)
        solar_irradiance: 0-1 normalized irradiance from weather model
        window_area_ratio: Fraction of wall area that is window (0.1-0.4)
        building_floor_area_m2: Building floor area

    Returns:
        Solar heat gain in kW
    """
    if solar_irradiance <= 0 or hour < 6 or hour >= 19:
        return 0.0

    # Estimated window area from floor area (assume 4m ceiling, 4 walls, approximate perimeter)
    perimeter_m = 4 * np.sqrt(building_floor_area_m2)
    wall_area_m2 = perimeter_m * 4.0  # 4m ceiling
    window_area_m2 = wall_area_m2 * window_area_ratio

    # Solar irradiance at ground ~ 1000 W/m² peak; SHGC for typical glass ~ 0.6
    SHGC = 0.6
    PEAK_IRRADIANCE_KW_M2 = 1.0

    gain_kw = window_area_m2 * SHGC * PEAK_IRRADIANCE_KW_M2 * solar_irradiance
    return float(round(gain_kw / 1000.0, 3))  # Convert W to kW (already in kW)


def calculate_energy_consumption(
    heater_on: int,
    lights_on: int,
    ac_on: int = 0,
    fan_speed: int = 0,
    fresh_air_damper: int = 0,
    heater_kw: float = 5.0,
    lights_kw: float = 0.5,
    ac_kw: float = 3.5,
) -> float:
    """
    Calculate total electrical energy consumption for the step (kWh).
    Step duration is 10 minutes = 1/6 hour.

    Returns:
        Energy in kWh for this step
    """
    # Fan uses proportional power
    fan_kw = {0: 0.0, 1: 0.1, 2: 0.25}.get(fan_speed, 0.0)

    # Fresh air damper fan motor
    damper_fan_kw = {0: 0.0, 1: 0.05, 2: 0.15, 3: 0.30}.get(fresh_air_damper, 0.0)

    total_kw = (
        heater_on * heater_kw
        + lights_on * lights_kw
        + ac_on * ac_kw
        + fan_kw
        + damper_fan_kw
    )

    return float(total_kw * DT_HOURS)  # kWh = kW × hours
