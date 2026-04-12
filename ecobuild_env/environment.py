"""
EcoBuild Core Simulation Environment.
Integrates all physics modules: thermal, weather, occupancy, humidity,
air quality, grid, solar/battery, genset, tariffs, and festivals.

Design: task_name is passed at reset() — same instance handles all 4 tasks.
All randomness is seeded through self.rng for episode reproducibility.
"""

import random
import uuid
import numpy as np
from typing import List, Optional, Tuple

from .models import BuildingObservation, BuildingAction, RewardBreakdown, EcoBuildState
from .thermal import update_temperature, calculate_solar_gain, calculate_energy_consumption
from .weather import IndianWeatherModel
from .occupancy import is_occupied_fixed, is_occupied_stochastic, get_predicted_occupancy
from .humidity import get_outdoor_humidity, update_indoor_humidity, humidity_comfort_penalty
from .air_quality import get_outdoor_aqi, update_indoor_aqi, update_indoor_co2, aqi_comfort_penalty
from .tariff import IndianElectricityTariff
from .grid import IndianGridModel
from .genset import DieselGenset
from .energy_sources import SolarPanelArray, BatteryStorage, EnergySourceController
from .festival_calendar import FestivalCalendar
from .task_configs import get_task_config, TaskConfig
from .tasks import StepData

# India grid CO2 factor
CO2_KG_PER_KWH = 0.82


class EcoBuildEnvironment:
    """
    Full EcoBuild simulation environment.
    Configured per-episode via reset(task_name, seed).
    """

    def __init__(self):
        """Initialize with no task — call reset() before use."""
        self.task_name: str = "basic_thermostat"
        self.cfg: Optional[TaskConfig] = None
        self.rng: Optional[np.random.RandomState] = None
        self.episode_id: str = ""
        self.episode_seed: Optional[int] = None

        # Simulation modules (initialized at reset)
        self.weather: Optional[IndianWeatherModel] = None
        self.tariff: Optional[IndianElectricityTariff] = None
        self.grid: Optional[IndianGridModel] = None
        self.genset: Optional[DieselGenset] = None
        self.energy_ctrl: Optional[EnergySourceController] = None
        self.festival_cal: Optional[FestivalCalendar] = None

        # State variables
        self.current_step: int = 0
        self.max_steps: int = 144
        self.hour_of_day: float = 0.0
        self.day_of_week: int = 0
        self.day_of_year: int = 180

        self.indoor_temp: float = 22.0
        self.indoor_humidity: float = 50.0
        self.indoor_aqi: float = 80.0
        self.indoor_co2: float = 420.0

        self.heater_status: int = 0
        self.ac_status: int = 0
        self.lights_status: int = 0
        self.fan_speed: int = 0
        self.fresh_air_damper: int = 0
        self.genset_status: int = 0
        self.battery_charge_rate: int = 0

        self.occupancy_count: int = 0

        # Accumulators
        self.total_energy_kwh: float = 0.0
        self.total_cost_inr: float = 0.0
        self.total_co2_kg: float = 0.0
        self.comfort_violations: int = 0

        self.episode_data: List[StepData] = []

    # ─────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────

    def reset(
        self,
        task_name: str = "basic_thermostat",
        seed: Optional[int] = None,
    ) -> BuildingObservation:
        """
        Reset environment to initial state for a new episode.

        Args:
            task_name: One of "basic_thermostat", "day_night_tou",
                       "load_shedding_optimizer", "multiday_optimization"
            seed: Random seed for reproducibility

        Returns:
            Initial BuildingObservation
        """
        self.task_name = task_name
        self.cfg = get_task_config(task_name)
        self.max_steps = self.cfg.duration_steps

        # Seeding
        self.episode_seed = seed
        self.rng = np.random.RandomState(seed)
        if seed is not None:
            random.seed(seed)

        self.episode_id = str(uuid.uuid4())
        self.current_step = 0

        # Time init
        self.day_of_week = int(self.rng.randint(0, 5))  # Weekday start
        self.hour_of_day = 0.0
        self.day_of_year = {
            "summer": 120, "monsoon": 200, "post_monsoon": 290, "winter": 15
        }.get(self.cfg.season, 120) + int(self.rng.randint(0, 10))

        # Init modules
        self.weather = IndianWeatherModel(
            region=self.cfg.region,
            season=self.cfg.season,
            rng=np.random.RandomState(seed if seed else 0),
        )
        self.tariff = IndianElectricityTariff(region=self.cfg.region)
        self.tariff.reset()
        self.grid = IndianGridModel(
            city=self.cfg.grid_city_profile,
            mode=self.cfg.grid_mode,
            rng=np.random.RandomState((seed or 0) + 1),
            total_steps=self.max_steps,
        )
        self.genset = DieselGenset(rng=np.random.RandomState((seed or 0) + 2)) \
            if self.cfg.has_genset else None
        if self.genset:
            self.genset.reset(rng=np.random.RandomState((seed or 0) + 2))

        solar = SolarPanelArray(rng=np.random.RandomState((seed or 0) + 3)) \
            if self.cfg.has_solar else SolarPanelArray(rated_kw=0)
        battery = BatteryStorage() if self.cfg.has_battery else BatteryStorage(capacity_kwh=0.01)
        self.energy_ctrl = EnergySourceController(
            solar=solar, battery=battery,
            has_solar=self.cfg.has_solar,
            has_battery=self.cfg.has_battery,
        )
        self.energy_ctrl.reset()

        self.festival_cal = FestivalCalendar(
            rng=np.random.RandomState((seed or 0) + 4)
        )

        # Physics init
        weather = self.weather.get_conditions(int(self.hour_of_day), self.day_of_year)
        self.indoor_temp = float(round(self.rng.uniform(
            self.cfg.comfort_temp_min, self.cfg.comfort_temp_max
        ), 1))
        self.indoor_humidity = 50.0
        self.indoor_aqi = 80.0
        self.indoor_co2 = 420.0

        # Actuator states
        self.heater_status = 0
        self.ac_status = 0
        self.lights_status = 0
        self.fan_speed = 0
        self.fresh_air_damper = 1  # Default: low fresh air
        self.genset_status = 0
        self.battery_charge_rate = 0

        self.occupancy_count = 0
        self.total_energy_kwh = 0.0
        self.total_cost_inr = 0.0
        self.total_co2_kg = 0.0
        self.comfort_violations = 0
        self.episode_data = []

        return self.get_observation()

    # ─────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────

    def step(self, action: BuildingAction) -> Tuple[BuildingObservation, float, bool, dict]:
        """
        Perform one simulation step (10 minutes).

        Args:
            action: BuildingAction with all actuator controls

        Returns:
            (observation, reward, done, info)
        """
        hour = int(self.hour_of_day)
        DT_HOURS = 1 / 6

        # 1. Apply actuator commands
        self.heater_status = int(action.heater_control)
        self.ac_status = int(action.ac_control)
        self.lights_status = int(action.lights_control)
        self.fan_speed = int(action.fan_speed)
        self.fresh_air_damper = int(action.fresh_air_damper)
        self.battery_charge_rate = int(action.battery_charge_rate)

        # 2. Grid state
        grid_state = self.grid.step(hour, self.cfg.season)
        grid_available = grid_state.is_available

        # 3. Force equipment off during power cuts (unless genset covers)
        genset_result = {"output_kw": 0.0, "cost_inr": 0.0}
        if not grid_available:
            genset_on = int(action.genset_control) if self.genset else 0
            if self.genset and genset_on:
                load_kw = (self.heater_status * 5.0 + self.ac_status * 3.5
                           + self.lights_status * 0.5)
                genset_result = self.genset.step(1, load_kw, DT_HOURS)
                if genset_result["output_kw"] < load_kw:
                    # Genset can't cover full load — partial shutdown
                    if self.ac_status and load_kw > genset_result["output_kw"]:
                        self.ac_status = 0
            else:
                # No backup: force all off
                self.heater_status = 0
                self.ac_status = 0
                self.lights_status = 0
        else:
            if self.genset:
                self.genset.step(0, 0.0, DT_HOURS)  # Keep genset state updated

        # 4. Weather update
        weather = self.weather.get_conditions(hour, self.day_of_year)

        # 5. Solar & battery dispatch
        load_kw = (self.heater_status * 5.0 + self.ac_status * 3.5
                   + self.lights_status * 0.5
                   + {0: 0, 1: 0.1, 2: 0.25}.get(self.fan_speed, 0)
                   + {0: 0, 1: 0.05, 2: 0.15, 3: 0.30}.get(self.fresh_air_damper, 0))

        tariff_rate = self.tariff.get_current_tariff(hour)
        energy_state = self.energy_ctrl.dispatch(
            load_kw=load_kw,
            hour=hour,
            day_of_year=self.day_of_year,
            season=self.cfg.season,
            grid_available=grid_available,
            tariff_rate=tariff_rate,
            battery_control=self.battery_charge_rate,
            dt_hours=DT_HOURS,
        )

        # 6. Energy accounting
        energy_kwh = calculate_energy_consumption(
            heater_on=self.heater_status,
            lights_on=self.lights_status,
            ac_on=self.ac_status,
            fan_speed=self.fan_speed,
            fresh_air_damper=self.fresh_air_damper,
        )
        grid_import_kwh = energy_state.grid_import_kw * DT_HOURS
        step_cost = self.tariff.get_step_cost(grid_import_kwh, load_kw, hour)
        step_cost += genset_result.get("cost_inr", 0.0)
        step_cost -= energy_state.net_metering_revenue_inr

        self.total_energy_kwh += energy_kwh
        self.total_cost_inr += step_cost
        co2_this_step = grid_import_kwh * CO2_KG_PER_KWH
        self.total_co2_kg += co2_this_step

        # 7. Festival calendar
        festival = self.festival_cal.get_multipliers(self.day_of_year)

        # 8. Occupancy update
        if self.task_name in ("basic_thermostat", "day_night_tou"):
            self.occupancy_count = is_occupied_fixed(
                hour, self.day_of_week, self.cfg.building_type, festival["occupancy_mult"]
            )
        else:
            self.occupancy_count = is_occupied_stochastic(
                hour, self.day_of_week, self.cfg.building_type,
                festival["occupancy_mult"], self.rng
            )
        occupied = self.occupancy_count > 0

        # 9. Thermal model (RC)
        solar_gain = calculate_solar_gain(hour, weather.solar_irradiance)
        equipment_damage = self.grid.equipment_damage_penalty(
            grid_state.voltage, bool(self.ac_status)
        ) if grid_available else 0.0

        self.indoor_temp = update_temperature(
            current_temp=self.indoor_temp,
            outdoor_temp=weather.temperature,
            heater_on=self.heater_status,
            ac_on=self.ac_status,
            occupancy_count=self.occupancy_count,
            solar_gain_kw=solar_gain,
            thermal_mass=self.cfg.thermal_mass,
            insulation=self.cfg.insulation,
        )

        # 10. Humidity update
        outdoor_hum = get_outdoor_humidity(hour, self.day_of_year, self.cfg.region, self.rng)
        self.indoor_humidity = update_indoor_humidity(
            self.indoor_humidity, outdoor_hum, self.ac_status, self.fan_speed, self.occupancy_count
        )

        # 11. Air quality update
        outdoor_aqi = get_outdoor_aqi(hour, self.day_of_year, self.cfg.region, self.rng)
        self.indoor_aqi = update_indoor_aqi(
            self.indoor_aqi, outdoor_aqi, self.fresh_air_damper, filtration_on=self.fan_speed > 0
        )
        self.indoor_co2 = update_indoor_co2(
            self.indoor_co2, self.occupancy_count, self.fresh_air_damper
        )

        # 12. Reward calculation
        reward_breakdown = self._calculate_reward(
            action=action,
            occupied=occupied,
            energy_cost_inr=step_cost,
            co2_this_step=co2_this_step,
            grid_available=grid_available,
            voltage=grid_state.voltage,
            equipment_damage=equipment_damage,
            net_metering_revenue=energy_state.net_metering_revenue_inr,
        )
        if occupied and not (self.cfg.comfort_temp_min <= self.indoor_temp <= self.cfg.comfort_temp_max):
            self.comfort_violations += 1

        # 13. Advance time
        self.current_step += 1
        self.hour_of_day = (self.current_step / 6) % 24
        if self.current_step % 144 == 0:
            self.day_of_week = (self.day_of_week + 1) % 7
            self.day_of_year = (self.day_of_year + 1) % 365

        done = self.current_step >= self.max_steps

        # 14. Record step data for grader
        self.episode_data.append(StepData(
            energy_kwh=energy_kwh,
            energy_cost_inr=step_cost,
            occupied=occupied,
            occupancy_count=self.occupancy_count,
            temp=self.indoor_temp,
            heater_on=bool(self.heater_status),
            ac_on=bool(self.ac_status),
            lights_on=bool(self.lights_status),
            grid_available=grid_available,
            genset_on=bool(genset_result.get("output_kw", 0) > 0),
            genset_cost_inr=genset_result.get("cost_inr", 0.0),
            solar_kw=energy_state.solar_generation_kw,
            humidity=self.indoor_humidity,
            indoor_aqi=self.indoor_aqi,
        ))

        info = {
            "reward_breakdown": reward_breakdown.model_dump(),
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "step": self.current_step,
            "co2_kg_this_step": round(co2_this_step, 4),
            "cumulative_co2_kg": round(self.total_co2_kg, 3),
            "energy_kwh_this_step": round(energy_kwh, 4),
            "total_energy_kwh": round(self.total_energy_kwh, 3),
            "total_cost_inr": round(self.total_cost_inr, 2),
            "grid_available": grid_available,
            "grid_voltage": round(grid_state.voltage, 1),
            "solar_kw": round(energy_state.solar_generation_kw, 2),
            "battery_soc_pct": round(energy_state.battery_soc_pct, 1),
            "festival": festival["festival_name"],
            "outdoor_aqi": round(outdoor_aqi, 1),
        }

        return self.get_observation(), float(reward_breakdown.total), done, info

    # ─────────────────────────────────────────────
    # REWARD
    # ─────────────────────────────────────────────

    def _calculate_reward(
        self,
        action: BuildingAction,
        occupied: bool,
        energy_cost_inr: float,
        co2_this_step: float,
        grid_available: bool,
        voltage: float,
        equipment_damage: float,
        net_metering_revenue: float,
    ) -> RewardBreakdown:
        """Compute multi-objective reward for this step."""
        # 1. Energy cost (normalized — ₹ cost as negative reward)
        energy_cost_penalty = -energy_cost_inr * 0.01

        # 2. Comfort penalty
        comfort_penalty = 0.0
        if occupied:
            if self.indoor_temp < self.cfg.comfort_temp_min:
                comfort_penalty = -(self.cfg.comfort_temp_min - self.indoor_temp) * 0.15
            elif self.indoor_temp > self.cfg.comfort_temp_max:
                comfort_penalty = -(self.indoor_temp - self.cfg.comfort_temp_max) * 0.15

        # 3. Humidity comfort
        hum_penalty = -humidity_comfort_penalty(self.indoor_humidity)

        # 4. AQI / CO2 penalty
        aq_penalty = -aqi_comfort_penalty(self.indoor_aqi, self.indoor_co2)

        # 5. Vacancy waste
        vacancy_waste = 0.0
        if not occupied:
            if action.heater_control:
                vacancy_waste -= 0.08
            if action.ac_control:
                vacancy_waste -= 0.08
            if action.lights_control:
                vacancy_waste -= 0.04

        # 6. Constraint violation (extreme temps — large penalty)
        constraint_violation = 0.0
        if self.indoor_temp < 16.0 or self.indoor_temp > 30.0:
            constraint_violation = -2.0

        # 7. Anticipation bonus (pre-heating before next-step occupancy)
        anticipation_bonus = 0.0
        next_hour = (int(self.hour_of_day) + 1) % 24
        next_occ = get_predicted_occupancy(
            next_hour, self.day_of_week, 1, self.cfg.building_type
        )
        if next_occ > 0 and not occupied:
            if action.heater_control and self.indoor_temp < self.cfg.comfort_temp_min:
                anticipation_bonus = 0.05

        # 8. Net metering bonus
        net_meter_bonus = net_metering_revenue * 0.005

        # 9. Equipment damage
        dmg_penalty = -equipment_damage * 0.1

        total = (
            energy_cost_penalty + comfort_penalty + hum_penalty + aq_penalty
            + vacancy_waste + constraint_violation + anticipation_bonus
            + net_meter_bonus + dmg_penalty
        )

        return RewardBreakdown(
            total=float(round(total, 4)),
            energy_cost_inr=float(round(energy_cost_inr, 3)),
            comfort_penalty=float(round(comfort_penalty, 4)),
            humidity_penalty=float(round(hum_penalty, 4)),
            aqi_penalty=float(round(aq_penalty, 4)),
            vacancy_waste=float(round(vacancy_waste, 4)),
            constraint_violation=float(round(constraint_violation, 4)),
            anticipation_bonus=float(round(anticipation_bonus, 4)),
            net_metering_revenue=float(round(net_metering_revenue, 4)),
            equipment_damage_penalty=float(round(dmg_penalty, 4)),
        )

    # ─────────────────────────────────────────────
    # OBSERVATION & STATE
    # ─────────────────────────────────────────────

    def get_observation(self) -> BuildingObservation:
        """Build and return the current observation."""
        hour = int(self.hour_of_day)
        weather = self.weather.get_conditions(hour, self.day_of_year) \
            if self.weather else None
        outdoor_hum = get_outdoor_humidity(hour, self.day_of_year, self.cfg.region if self.cfg else "generic")
        outdoor_aqi = get_outdoor_aqi(hour, self.day_of_year, self.cfg.region if self.cfg else "generic")

        grid_state = self.grid.step(max(0, self.current_step - 1), self.cfg.season if self.cfg else "summer") \
            if self.grid else None

        genset_state = self.genset.get_state() if self.genset else None
        energy_state = self.energy_ctrl.solar.get_generation(hour, self.day_of_year) \
            if self.energy_ctrl and self.energy_ctrl.has_solar else 0.0
        battery_soc = self.energy_ctrl.battery.soc_pct if self.energy_ctrl else 0.0

        festival = self.festival_cal.get_multipliers(self.day_of_year) \
            if self.festival_cal else {"days_until_next_festival": 99, "occupancy_mult": 1.0}

        return BuildingObservation(
            indoor_temperature=round(self.indoor_temp, 2),
            outdoor_temperature=round(weather.temperature if weather else 15.0, 2),
            humidity=round(self.indoor_humidity, 1),
            outdoor_humidity=round(outdoor_hum, 1),
            occupancy_count=int(self.occupancy_count),
            predicted_occupancy_2h=get_predicted_occupancy(
                hour, self.day_of_week, 2, self.cfg.building_type if self.cfg else "office"
            ),
            hour_of_day=hour,
            day_of_week=int(self.day_of_week),
            time_step=int(self.current_step),
            heater_status=int(self.heater_status),
            ac_status=int(self.ac_status),
            lights_status=int(self.lights_status),
            fan_speed=int(self.fan_speed),
            fresh_air_damper=int(self.fresh_air_damper),
            cumulative_energy_kwh=round(self.total_energy_kwh, 3),
            electricity_price_normalized=self.tariff.get_normalized_tariff(hour) if self.tariff else 0.5,
            grid_available=int(grid_state.is_available) if grid_state else 1,
            grid_voltage=round(grid_state.voltage if grid_state else 220.0, 1),
            predicted_next_cut_minutes=grid_state.predicted_next_cut_minutes if grid_state else 999,
            solar_generation_kw=round(energy_state, 2),
            battery_soc_pct=round(battery_soc, 1),
            genset_available=int(genset_state.fuel_level_liters > 10) if genset_state else 0,
            genset_fuel_pct=round(genset_state.fuel_level_liters / self.genset.tank_capacity * 100
                                  if genset_state and self.genset else 0.0, 1),
            outdoor_aqi=round(outdoor_aqi, 1),
            indoor_aqi=round(self.indoor_aqi, 1),
            indoor_co2_ppm=round(self.indoor_co2, 0),
            days_until_next_festival=self.festival_cal.days_until_next_festival(self.day_of_year)
                if self.festival_cal else 99,
            festival_occupancy_mult=festival.get("occupancy_mult", 1.0),
        )

    def state(self) -> EcoBuildState:
        """Return episode-level state as typed Pydantic model."""
        from .tasks import evaluate_episode
        score = 0.0
        if self.episode_data:
            try:
                score = evaluate_episode(self.task_name, self.episode_data)
            except Exception:
                score = 0.0

        return EcoBuildState(
            episode_id=self.episode_id,
            task_name=self.task_name,
            step_count=self.current_step,
            max_steps=self.max_steps,
            total_energy_kwh=round(self.total_energy_kwh, 3),
            total_cost_inr=round(self.total_cost_inr, 2),
            comfort_violations=self.comfort_violations,
            is_done=self.current_step >= self.max_steps,
            episode_seed=self.episode_seed,
            current_score=round(score, 4),
            season=self.cfg.season if self.cfg else "summer",
            region=self.cfg.region if self.cfg else "generic",
        )

    def grade(self) -> float:
        """Grade the current episode. Returns score in [0.0, 1.0]."""
        from .tasks import evaluate_episode
        if not self.episode_data:
            return 0.0
        return evaluate_episode(self.task_name, self.episode_data)
