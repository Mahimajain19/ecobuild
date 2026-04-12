"""
EcoBuild Inference Runner.
Supports baseline rule-based agent and LLM agent via OpenAI-compatible API.
Reads config from scenario_config.json and saves episode results to outputs/evals/.
"""

import os
import sys
import json
import logging
from typing import List, Optional
from pathlib import Path
from datetime import datetime

from ecobuild_env.environment import EcoBuildEnvironment
from ecobuild_env.models import BuildingAction, BuildingObservation
from ecobuild_env.tasks import StepData, evaluate_episode, evaluate_episode_breakdown

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] ecobuild | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ecobuild")


# ─────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────

class BaselineAgent:
    """
    Rule-based agent: simple thermostat logic.
    Heater on when cold, AC on when hot and occupied, lights on when occupied.
    Turns off all equipment when vacant.
    """

    def get_action(self, obs: BuildingObservation, comfort_min: float = 21.0, comfort_max: float = 24.0) -> BuildingAction:
        occupied = obs.occupancy_count > 0
        heater = 1 if obs.indoor_temperature < comfort_min and not obs.ac_status else 0
        ac = 1 if obs.indoor_temperature > comfort_max and occupied else 0
        lights = 1 if occupied else 0
        fan = 1 if occupied and obs.indoor_co2_ppm > 800 else 0
        damper = 1 if obs.indoor_co2_ppm > 1000 else 0
        genset = 1 if not obs.grid_available and occupied else 0
        return BuildingAction(
            heater_control=heater,
            ac_control=ac,
            lights_control=lights,
            fan_speed=fan,
            fresh_air_damper=damper,
            genset_control=genset,
            battery_charge_rate=0,
        )


class OpenAIAgent:
    """LLM-based agent using OpenAI-compatible API."""

    SYSTEM_PROMPT = (
        "You are an intelligent building energy management system controlling HVAC and lighting. "
        "Your goal: minimize energy cost while maintaining occupant comfort (stay in comfort zone). "
        "During power cuts, use genset only for critical loads. Pre-cool/pre-heat before occupancy. "
        "Respond ONLY with valid JSON matching the action schema."
    )

    def __init__(self, api_base_url: str, model_name: str, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model_name = model_name

    def get_action(self, obs: BuildingObservation, comfort_min: float = 21.0, comfort_max: float = 24.0) -> BuildingAction:
        state_summary = (
            f"Indoor: {obs.indoor_temperature}°C (target {comfort_min}-{comfort_max}°C), "
            f"Humidity: {obs.humidity}%, "
            f"Occupants: {obs.occupancy_count} (predicted in 2h: {obs.predicted_occupancy_2h}), "
            f"Hour: {obs.hour_of_day}:00, "
            f"Grid: {'ON' if obs.grid_available else 'OFF (POWER CUT)'}, "
            f"Voltage: {obs.grid_voltage}V, "
            f"Tariff: {'peak ₹8/kWh (18-22h)' if 18 <= obs.hour_of_day < 22 else 'normal/off-peak'}, "
            f"Next cut in: {obs.predicted_next_cut_minutes} min, "
            f"Solar: {obs.solar_generation_kw}kW, Battery: {obs.battery_soc_pct}%, "
            f"Outdoor AQI: {obs.outdoor_aqi:.0f}, Indoor CO2: {obs.indoor_co2_ppm:.0f}ppm, "
            f"Festival mult: {obs.festival_occupancy_mult}x."
        )

        prompt = (
            f"Current State: {state_summary}\n\n"
            f"Decide the next action. Return JSON with keys:\n"
            f"heater_control (0/1), ac_control (0/1), lights_control (0/1), "
            f"fan_speed (0/1/2), fresh_air_damper (0/1/2/3), "
            f"genset_control (0/1 — only use if grid=OFF), battery_charge_rate (0/1/2)"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                timeout=15.0,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content.strip())
            return BuildingAction(
                heater_control=int(data.get("heater_control", 0)),
                ac_control=int(data.get("ac_control", 0)),
                lights_control=int(data.get("lights_control", 0)),
                fan_speed=int(data.get("fan_speed", 0)),
                fresh_air_damper=int(data.get("fresh_air_damper", 1)),
                genset_control=int(data.get("genset_control", 0)),
                battery_charge_rate=int(data.get("battery_charge_rate", 0)),
            )
        except Exception as e:
            logger.warning(f"LLM action parse failed: {e} — using fallback")
            return BuildingAction(heater_control=0, ac_control=0, lights_control=0)


# ─────────────────────────────────────────────
# Output Saving
# ─────────────────────────────────────────────

def save_episode_results(task_name: str, final_score: float, rewards: List[float],
                         episode_data: List[StepData], seed: Optional[int], model_name: str):
    """Save episode results to outputs/evals/ as JSON."""
    out_dir = Path("outputs/evals")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = out_dir / f"{task_name}_{timestamp}.json"

    try:
        breakdown = evaluate_episode_breakdown(task_name, episode_data)
    except Exception:
        breakdown = {}

    result = {
        "task_name": task_name,
        "timestamp": timestamp,
        "model": model_name,
        "seed": seed,
        "final_score": round(final_score, 4),
        "num_steps": len(rewards),
        "total_reward": round(sum(rewards), 4),
        "mean_reward": round(sum(rewards) / max(1, len(rewards)), 4),
        "min_reward": round(min(rewards) if rewards else 0, 4),
        "max_reward": round(max(rewards) if rewards else 0, 4),
        "score_breakdown": breakdown,
    }

    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Episode saved → {fname}")


# ─────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────

def run_inference(task_name: str, config: dict):
    """Run one full episode for a given task and config."""
    API_BASE_URL = config.get("llm_api_base", os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"))
    MODEL_NAME = config.get("llm_model", os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"))
    HF_TOKEN = config.get("llm_api_key", os.getenv("HF_TOKEN"))
    SEED = config.get("seed", None)
    COMFORT_MIN = config.get("comfort_range", [21.0, 24.0])[0]
    COMFORT_MAX = config.get("comfort_range", [21.0, 24.0])[1]

    # Choose agent
    if MODEL_NAME != "baseline-rule-based" and HF_TOKEN:
        try:
            agent = OpenAIAgent(API_BASE_URL, MODEL_NAME, HF_TOKEN)
            logger.info(f"Using LLM agent: {MODEL_NAME}")
        except Exception as e:
            logger.warning(f"LLM agent init failed ({e}), falling back to rule-based")
            MODEL_NAME = "baseline-rule-based"
            agent = BaselineAgent()
    else:
        MODEL_NAME = "baseline-rule-based"
        agent = BaselineAgent()

    logger.info(f"START task={task_name} model={MODEL_NAME} seed={SEED}")

    env = EcoBuildEnvironment()
    obs = env.reset(task_name=task_name, seed=SEED)

    done = False
    step_n = 0
    rewards: List[float] = []
    success = True

    while not done:
        try:
            action = agent.get_action(obs, COMFORT_MIN, COMFORT_MAX)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            logger.info(
                f"step={step_n} "
                f"temp={obs.indoor_temperature:.1f}°C "
                f"occ={obs.occupancy_count} "
                f"reward={reward:.3f} "
                f"done={done} "
                f"cost=₹{info.get('total_cost_inr', 0):.1f}"
            )
        except Exception as e:
            logger.error(f"Step error at step {step_n}: {e}")
            success = False
            done = True
        step_n += 1

    # Grade
    final_score = 0.0
    try:
        final_score = env.grade()
    except Exception as e:
        logger.error(f"Grading error: {e}")

    rewards_summary = ",".join(f"{r:.3f}" for r in rewards[:5]) + ("..." if len(rewards) > 5 else "")
    logger.info(
        f"END success={success} steps={step_n} score={final_score:.4f} "
        f"total_cost=₹{env.total_cost_inr:.1f} "
        f"co2={env.total_co2_kg:.2f}kg "
        f"rewards_sample=[{rewards_summary}]"
    )

    save_episode_results(task_name, final_score, rewards, env.episode_data, SEED, MODEL_NAME)
    return final_score


def load_config() -> dict:
    """Load scenario_config.json if present, else return empty dict."""
    config_path = Path("scenario_config.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    config = load_config()
    # Run the task specified in config, or default to all tasks
    task_override = config.get("task_name", None)

    if task_override:
        tasks_to_run = [task_override]
    else:
        tasks_to_run = ["basic_thermostat", "day_night_tou", "load_shedding_optimizer"]

    for task in tasks_to_run:
        run_inference(task, config)
