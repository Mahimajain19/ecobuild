from dataclasses import dataclass
from typing import List

# Baselines for energy scores
BASELINE_ALWAYS_ON = 5.0 
BASELINE_SIMPLE_THERMOSTAT = 2.5
BASELINE_SMART_THERMOSTAT = 1.5

@dataclass
class StepData:
    energy: float
    occupied: bool
    temp: float
    heater_on: bool
    lights_on: bool

def grade_task1(episode_data: List[StepData]) -> float:
    """
    Grader for Task 1: EASY - Basic Thermostat Control
    """
    if not episode_data:
        return 0.0
        
    energy_used = sum(step.energy for step in episode_data)
    baseline_energy = BASELINE_ALWAYS_ON * len(episode_data)
    energy_score = max(0.0, 1.0 - energy_used / baseline_energy) if baseline_energy > 0 else 0.0
    
    comfort_violations = sum(
        1 for step in episode_data 
        if step.occupied and not (20.0 <= step.temp <= 22.0)
    )
    comfort_score = max(0.0, 1.0 - comfort_violations / 10.0)
    
    return float(0.6 * energy_score + 0.4 * comfort_score)

def grade_task2(episode_data: List[StepData]) -> float:
    """
    Grader for Task 2: MEDIUM - Multi-Actuator with Day/Night Cycle
    """
    if not episode_data:
        return 0.0
        
    # Energy efficiency
    energy_used = sum(step.energy for step in episode_data)
    baseline_energy = BASELINE_SIMPLE_THERMOSTAT * len(episode_data)
    energy_score = max(0.0, 1.0 - energy_used / baseline_energy) if baseline_energy > 0 else 0.0
    
    # Comfort compliance
    comfort_violations = sum(
        1 for step in episode_data 
        if step.occupied and not (20.0 <= step.temp <= 22.0)
    )
    comfort_score = max(0.0, 1.0 - comfort_violations / 20.0)
    
    # Lighting efficiency (penalize lights on when vacant)
    lighting_waste = sum(
        1 for step in episode_data 
        if (not step.occupied) and step.lights_on
    )
    lighting_score = max(0.0, 1.0 - lighting_waste / 30.0)
    
    return float(0.5 * energy_score + 0.3 * comfort_score + 0.2 * lighting_score)

def calculate_anticipation_bonus(episode_data: List[StepData]) -> float:
    anticipation_score = 0.0
    for i in range(1, len(episode_data)):
        if episode_data[i].occupied and not episode_data[i-1].occupied:
            if episode_data[i-1].heater_on:
                anticipation_score += 1.0
    return min(1.0, anticipation_score / 5.0)

def grade_task3(episode_data: List[StepData]) -> float:
    """
    Grader for Task 3: HARD - Multi-Day Optimization with Weather Uncertainty
    """
    if not episode_data:
        return 0.0
        
    # Advanced energy efficiency
    energy_used = sum(step.energy for step in episode_data)
    baseline_energy = BASELINE_SMART_THERMOSTAT * len(episode_data)
    energy_saved_pct = max(0.0, (baseline_energy - energy_used) / baseline_energy) if baseline_energy > 0 else 0.0
    energy_score = min(1.0, energy_saved_pct * 2.0)  # 50% savings = full score
    
    # Strict comfort compliance
    comfort_violations = sum(
        1 for step in episode_data 
        if step.occupied and not (20.0 <= step.temp <= 22.0)
    )
    total_occupied = sum(1 for step in episode_data if step.occupied)
    
    if total_occupied > 0:
        comfort_score = max(0.0, 1.0 - comfort_violations / (total_occupied * 0.1))
    else:
        comfort_score = 1.0
        
    # Anticipatory heating (bonus for pre-heating before occupancy)
    anticipation_score = calculate_anticipation_bonus(episode_data)
    
    # Penalty for energy waste during vacancy
    vacancy_heating = sum(
        1 for step in episode_data if (not step.occupied) and step.heater_on
    )
    vacancy_score = max(0.0, 1.0 - vacancy_heating / (len(episode_data) * 0.2)) if len(episode_data) > 0 else 1.0
    
    return float(
        0.4 * energy_score + 
        0.35 * comfort_score + 
        0.15 * anticipation_score + 
        0.1 * vacancy_score
    )

def evaluate_episode(task_name: str, episode_data: List[StepData]) -> float:
    """
    Evaluate an episode based on the task name.
    """
    if task_name == "basic_thermostat":
        return grade_task1(episode_data)
    elif task_name == "day_night_control":
        return grade_task2(episode_data)
    elif task_name == "multiday_optimization":
        return grade_task3(episode_data)
    else:
        raise ValueError(f"Unknown task name: {task_name}")
