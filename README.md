<div align="center">

# ⚡ EcoBuild
**A Smart Building Energy Management RL Environment for India**

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen?style=for-the-badge)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Server-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License MIT](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

*Optimize comfort, mitigate power cuts, and slash energy costs in a highly realistic Indian smart building simulation.*

</div>

---

## 📖 Overview

**EcoBuild** is a highly realistic, production-ready Reinforcement Learning (RL) and LLM agent environment simulating a commercial building. Built for the **OpenEnv** framework, it goes beyond generic Western HVAC simulators by baking in the complex, dynamic realities of the **Indian power grid and climate**. 

Agents must learn to balance human comfort (Temperature, Humidity, AQI) with energy efficiency, while navigating extreme weather, Time-of-Use tariffs, and unpredictable power cuts.

***

## ✨ Key Differentiators: The "India Context"

EcoBuild isn't just an RC thermal model; it models the constraints of developing-world infrastructure:

*   🔌 **Grid Reliability & Load Shedding:** Scheduled and stochastic power cuts (e.g., 5 hrs/day in Tier-2 cities). Agents must pre-condition buildings before outages.
*   💨 **Air Quality (AQI) & CO₂ Trade-offs:** Delhi winters face hazardous AQI (300-500). The agent decides when to open the fresh air damper (lowers CO₂, but raises PM2.5) vs. running expensive HEPA filtration.
*   🌦️ **Regional Asymmetric Seasons:** Realistic weather profiles for Delhi, Mumbai, and Bangalore. From Mumbai's 90% monsoon humidity (requiring AC dehumidification) to Delhi's extreme winter/summer delta.
*   🪫 **Tri-Source Energy Dispatch:** Seamlessly switch between **Rooftop Solar** (free), **Lithium Battery** (stored), **Grid Supply** (Time-of-Use Tariffs ₹3-₹8/kWh), and a **Diesel Genset** (expensive backup at ₹35/kWh).
*   🪔 **Festival Calendar:** Predictable load spikes. Massive lighting and double-occupancy during Diwali or Navratri.

***

## ⚙️ Physics & Dynamics

EcoBuild employs sophisticated, non-linear physics engines for realistic building dynamics:
*   **RC Thermal Model:** Models thermal mass (capacitance) and insulation (resistance). Heating/cooling takes time, requiring anticipation. Window solar heat gain is modeled dynamically.
*   **Poisson Occupancy:** People arrive and leave based on realistic office/residential profiles with stochastic noise. Each person contributes 80W of sensible heat and moisture.
*   **Multi-Actuator Control:** Continuous and binary controls for Heater, AC, Lights, Fan Speed, Fresh Air Damper, Genset, and Battery Storage.

***

## 🎯 Task Suite

The environment ships with 4 progressively difficult tasks designed to test RL and LLM agents:

| Task Name | Difficulty | Length | Focus Area |
| :--- | :---: | :---: | :--- |
| `basic_thermostat` | **Easy** | 24 Hours | Delhi winter. Learn thermal mass lag and pre-heating before morning office arrivals. |
| `day_night_tou` | **Medium** | 24 Hours | Delhi summer. Shift cooling loads to avoid exorbitant peak evening tariffs (6 PM - 10 PM). |
| `load_shedding_optimizer` | **Hard** | 48 Hours | Mumbai monsoon. scheduled 5h/day power cuts. Manage expensive diesel gensets and pre-condition for outages. |
| `multiday_optimization` | **Expert** | 7 Days | Full system optimization: Solar + Battery + TOU + Smog Event (Day 6) + Festival Spike (Day 4). |

***

## 💻 API & Spaces

Fully compliant with `meta-pytorch/OpenEnv`.

### Observation Space (28 Features)
*   **Thermal/Air:** `indoor_temperature`, `outdoor_temperature`, `humidity`, `outdoor_aqi`, `indoor_co2_ppm`
*   **Energy/Grid:** `grid_voltage`, `grid_available`, `predicted_next_cut_minutes`, `electricity_price_normalized`
*   **Sources:** `solar_generation_kw`, `battery_soc_pct`, `genset_fuel_pct`
*   **Context:** `occupancy_count`, `hour_of_day`, `days_until_next_festival`

### Action Space (7 Actuators)
*   `heater_control` (0/1)
*   `ac_control` (0/1)
*   `lights_control` (0/1)
*   `fan_speed` (0, 1, 2)
*   `fresh_air_damper` (0, 15%, 50%, 100%)
*   `genset_control` (0/1 - manual override)
*   `battery_charge_rate` (Auto, Force Charge, Force Discharge)

***

## 🚀 Installation & Quickstart

### 1. Local Development
Requires Python 3.10+.
```bash
git clone https://github.com/yourusername/ecobuild.git
cd ecobuild
pip install -r requirements.txt
pip install -e .[dev]
```

### 2. Running the Standard OpenEnv Server
EcoBuild runs as a FastAPI HTTP/WebSocket server.
```bash
python server/app.py
```
> Server runs on `http://localhost:8000`. Features standard `/health`, `/ws`, `/tasks`, and `/grade` endpoints.

### 3. Docker Deployment
Production-ready Dockerfile included with non-root security and healthchecks.
```bash
docker build -t ecobuild-server .
docker run -p 8000:8000 ecobuild-server
```

***

## 🤖 Evaluating LLM & RL Agents

EcoBuild natively supports evaluating agents. We provide a built-in rule-based baseline and an OpenAI-compatible LLM agent runner.

Edit `scenario_config.json` to configure the LLM provider (e.g., Hugging Face, OpenAI, vLLM).

```bash
# Run baseline and LLM inference
python inference.py
```

Results and detailed grader breakdowns are saved to `outputs/evals/`.

***

## 🏗️ Architecture

```text
ecobuild/
├── ecobuild_env/        # Core Simulation & Physics Engine
│   ├── air_quality.py   # PM2.5 / AQI / CO2 logic
│   ├── energy_sources.py# Solar panels & Li-ion Battery
│   ├── environment.py   # Main Environment Integrator
│   ├── grid.py          # Load shedding & voltage drops
│   ├── tariff.py        # TOU Pricing models
│   ├── tasks.py         # Calibrated Grader classes
│   ├── thermal.py       # RC Circuit Thermal Mass modeling
│   └── weather.py       # Regional seasonal profiles
├── server/
│   └── app.py           # FastAPI + WebSockets Server
├── tests/               # Comprehensive Pytest suite
├── inference.py         # LLM/Baseline agent evaluation runner
├── openenv.yaml         # OpenEnv framework manifest
└── scenario_config.json # Runtime configuration
```

***

## 🧪 Testing

Run the comprehensive test suite validating physics, constraints, and tasks:
```bash
pytest tests/ -v
```

***

<div align="center">
<i>Built for the next generation of real-world AI building control.</i>
</div>
