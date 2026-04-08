import random
import numpy as np
from .models import BuildingObservation, BuildingAction, BuildingReward
from .thermal import update_temperature, calculate_energy_consumption
from .weather import get_outdoor_temp
from .occupancy import is_occupied_fixed, is_occupied_stochastic

class EcoBuildEnv:
    def __init__(self, task_name="basic_thermostat"):
        self.task_name = task_name
        self.max_steps = 144  # 24 hours @ 10 min per step
        self.reset()
        
    def reset(self):
        """Reset the environment to initial conditions."""
        self.current_step = 0
        self.day_of_week = random.randint(0, 4) if self.task_name == "basic_thermostat" else random.randint(0, 6)
        self.hour_of_day = 0
        
        # Initial indoor temperature: between 18-22°C
        self.indoor_temp = float(round(random.uniform(18.0, 22.0), 2))
        self.outdoor_temp = float(round(get_outdoor_temp(self.hour_of_day, 0), 2))
        
        self.heater_status = 0
        self.lights_status = 0
        
        # Initial occupancy
        self.occupancy = self._get_occupancy()
        
        return self.get_observation()
        
    def _get_occupancy(self):
        if self.task_name == "multiday_optimization":
            return is_occupied_stochastic(self.hour_of_day, self.day_of_week)
        return is_occupied_fixed(self.hour_of_day, self.day_of_week)
        
    def get_observation(self):
        """Return the current building observation."""
        return BuildingObservation(
            indoor_temperature=self.indoor_temp,
            outdoor_temperature=self.outdoor_temp,
            occupancy=self.occupancy,
            hour_of_day=int(self.hour_of_day),
            day_of_week=self.day_of_week,
            heater_status=self.heater_status,
            lights_status=self.lights_status,
            time_step=self.current_step
        )
        
    def state(self):
        """Return the current state as a dictionary."""
        return self.get_observation().model_dump()
        
    def step(self, action: BuildingAction):
        """Perform one simulation step."""
        # 1. Update internal actuators
        self.heater_status = action.heater_control
        self.lights_status = action.lights_control
        
        # 2. Calculate energy and reward (placeholder logic, will refine in rewards.py or inline)
        energy_used = calculate_energy_consumption(self.heater_status, self.lights_status)
        
        # 3. Simulate Environment Dynamics
        # 10 minute time step
        self.current_step += 1
        self.hour_of_day = (self.current_step / 6) % 24
        
        # Update outdoor temp
        self.outdoor_temp = get_outdoor_temp(int(self.hour_of_day), self.day_of_week)
        
        # Update indoor temp
        self.indoor_temp = update_temperature(
            self.indoor_temp, 
            self.outdoor_temp, 
            self.heater_status
        )
        
        # Update occupancy
        self.occupancy = self._get_occupancy()
        
        # 4. Calculate Reward
        reward_info = self._calculate_reward(action, energy_used)
        
        # 5. Check termination
        done = self.current_step >= self.max_steps
        if self.task_name == "multiday_optimization":
             # Extend for multi-day in Task 3? 
             # Let's keep it simple for now and handle 7 days elsewhere or via max_steps
             if self.current_step >= 144 * 2:
                  done = True
        
        return self.get_observation(), reward_info.reward, done, reward_info.model_dump()
        
    def _calculate_reward(self, action, energy_used):
        # Constants for reward balancing
        HEATER_WEIGHT = 0.5
        LIGHTS_WEIGHT = 0.05
        COMFORT_WEIGHT = 2.0
        VACANCY_PENALTY_WEIGHT = 0.5
        
        # Energy Cost (negative reward)
        energy_score = -(action.heater_control * HEATER_WEIGHT + action.lights_control * LIGHTS_WEIGHT)
        
        # Comfort Penalty (only if occupied)
        comfort_penalty = 0.0
        if self.occupancy == 1:
            if self.indoor_temp < 20.0:
                comfort_penalty = (20.0 - self.indoor_temp) * COMFORT_WEIGHT
            elif self.indoor_temp > 22.0:
                comfort_penalty = (self.indoor_temp - 22.0) * COMFORT_WEIGHT
        
        # Vacancy Waste (negative bonus)
        vacancy_waste = 0.0
        if self.occupancy == 0:
            if action.heater_control == 1:
                vacancy_waste = 1.0 * VACANCY_PENALTY_WEIGHT
            if action.lights_control == 1:
                 vacancy_waste += 0.5 * VACANCY_PENALTY_WEIGHT
                 
        total_reward = float(energy_score - comfort_penalty - vacancy_waste)
        
        # Normalize to roughly [-1, 0] or [0, 1] - let's stay in negative-offset for costs
        normalized_reward = total_reward / 10.0
        
        return BuildingReward(
            reward=normalized_reward,
            energy_cost=energy_score,
            comfort_penalty=comfort_penalty,
            vacancy_waste=vacancy_waste
        )
