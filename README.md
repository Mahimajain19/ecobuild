---
title: EcoBuild Energy Env
emoji: 🏢
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
tags:
- openenv
pinned: false
---

# EcoBuild: Smart Building Energy Management RL Environment

## Motivation
Buildings account for roughly 30% of global energy consumption. The HVAC (Heating, Ventilation, and Air Conditioning) systems within them are major contributors to CO₂ emissions. Developing intelligent building energy management controls can drastically reduce wasted energy without sacrificing human comfort.

**EcoBuild** is an OpenEnv-compliant RL environment where agents act as intelligent thermostats, learning to balance critical real-world energy-efficiency goals against strict building regulations and occupant comfort constraints.

## Technical Scope

### Observation Space
The state is derived dynamically per step to represent the building's context.

| Variable | Type | Description |
|---|---|---|
| `indoor_temperature` | float | Current indoor temperature (°C). |
| `outdoor_temperature` | float | Current outdoor temperature simulating sin-curved weather cycles. |
| `occupancy` | integer | Boolean indicator (0 = vacant, 1 = occupied). |
| `hour_of_day` | integer | Hour of the respective day [0-23]. |
| `day_of_week` | integer | Day of the respective week [0-6] (Mon=0). |
| `heater_status` | integer | Current state of heater (0 = off, 1 = on). |
| `lights_status` | integer | Current state of lights (0 = off, 1 = on). |
| `time_step` | integer | The discrete simulation step. |

### Action Space
Agents are responsible for manipulating two actuators:
1. `heater_control`: integer [0, 1] mapped to toggling the thermal heater.
2. `lights_control`: integer [0, 1] mapped to toggling building lighting.

## Setup and Quick Action

1. Install environment requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. For testing inference:
   ```bash
   python inference.py
   ```

## Task Information

1. **`basic_thermostat` (EASY)**
   - Objective: Maintain temperature 20-22°C rigidly during occupied hours.
   - Context: Fixed outdoor temp and predictable weekday schedules.

2. **`day_night_control` (MEDIUM)** 
   - Objective: Control multiple actuators against rigid weather fluctuations.
   - Context: Predictable weather cycling over the period of a day enforcing lighting checks.

3. **`multiday_optimization` (HARD)**
   - Objective: Advance multi-day scheduling via anticipation checks. 
   - Context: Requires weather predictability buffering and penalty scaling over a 2-day episode (48 hours).

## Docker Setup
```bash
docker build -t ecobuild .
docker run -p 8000:8000 ecobuild
```
