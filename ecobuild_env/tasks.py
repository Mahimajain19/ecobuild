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
    
    total_occupied = sum(1 for step in episode_data if step.occupied)
    comfort_violations = sum(
        1 for step in episode_data 
        if step.occupied and not (20.0 <= step.temp <= 22.0)
    )
    comfort_score = max(0.0, 1.0 - comfort_violations / max(1.0, float(total_occupied)))
    
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
    total_occupied = sum(1 for step in episode_data if step.occupied)
    comfort_violations = sum(
        1 for step in episode_data 
        if step.occupied and not (20.0 <= step.temp <= 22.0)
    )
    comfort_score = max(0.0, 1.0 - comfort_violations / max(1.0, float(total_occupied)))
    
    # Lighting efficiency (penalize lights on when vacant)
    total_unoccupied = sum(1 for step in episode_data if not step.occupied)
    lighting_waste = sum(
        1 for step in episode_data 
        if (not step.occupied) and step.lights_on
    )
    lighting_score = max(0.0, 1.0 - lighting_waste / max(1.0, float(total_unoccupied)))
    
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
    max_energy = BASELINE_ALWAYS_ON * len(episode_data)
    energy_score = max(0.0, 1.0 - energy_used / max_energy) if max_energy > 0 else 0.0
    
    # Strict comfort compliance
    total_occupied = sum(1 for step in episode_data if step.occupied)
    comfort_violations = sum(
        1 for step in episode_data 
        if step.occupied and not (20.0 <= step.temp <= 22.0)
    )
    
    if total_occupied > 0:
        comfort_score = max(0.0, 1.0 - comfort_violations / float(total_occupied))
    else:
        comfort_score = 1.0
        
    # Anticipatory heating (bonus for pre-heating before occupancy)
    anticipation_score = calculate_anticipation_bonus(episode_data)
    
    # Penalty for energy waste during vacancy
    total_unoccupied = sum(1 for step in episode_data if not step.occupied)
    vacancy_heating = sum(
        1 for step in episode_data if (not step.occupied) and step.heater_on
    )
    vacancy_score = max(0.0, 1.0 - vacancy_heating / max(1.0, float(total_unoccupied))) if len(episode_data) > 0 else 1.0
    
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
