import os
import sys
import json
from typing import List
from ecobuild_env.environment import EcoBuildEnv
from ecobuild_env.models import BuildingAction, BuildingObservation
from ecobuild_env.tasks import StepData, evaluate_episode

class BaselineAgent:
    def get_action(self, obs: BuildingObservation) -> BuildingAction:
        return BuildingAction(
            heater_control=1 if obs.indoor_temperature < 20.0 else 0,
            lights_control=1 if obs.occupancy == 1 else 0
        )

class OpenAIAgent:
    def __init__(self, api_base_url, model_name, api_key):
        from openai import OpenAI
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model_name = model_name

    def get_action(self, obs: BuildingObservation) -> BuildingAction:
        prompt = (f"Current State: Temp {obs.indoor_temperature}C, Occupied: {obs.occupancy}, "
                  f"Hour: {obs.hour_of_day}. Maintain 20-22C when occupied. Decide action as JSON with "
                  f"'heater_control' (0 or 1) and 'lights_control' (0 or 1).")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an HVAC/lighting controller. Output perfectly valid JSON matching the requested structure."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                timeout=10.0  # 10s timeout per action
            )
            content = response.choices[0].message.content.strip()
            
            # Clean possible markdown formatting
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            data = json.loads(content)
            return BuildingAction(
                heater_control=int(data.get("heater_control", 0)),
                lights_control=int(data.get("lights_control", 0))
            )
        except Exception as e:
            # Fallback pattern on API or parse error
            return BuildingAction(heater_control=0, lights_control=0)

def run_inference(task_name: str):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Configure agent type based on environment parameters
    if MODEL_NAME != "baseline-rule-based" and API_BASE_URL and HF_TOKEN:
        try:
            agent = OpenAIAgent(API_BASE_URL, MODEL_NAME, HF_TOKEN)
        except ImportError:
            MODEL_NAME = "baseline-rule-based (OpenAI missing)"
            agent = BaselineAgent()
    else:
        MODEL_NAME = "baseline-rule-based"
        agent = BaselineAgent()

    print(f"[START] task={task_name} env=ecobuild model={MODEL_NAME}")
    sys.stdout.flush()

    env = EcoBuildEnv(task_name=task_name)
    obs = env.reset()
    
    done = False
    step_n = 0
    rewards: List[float] = []
    episode_data: List[StepData] = []
    
    success = True
    final_score = 0.0

    while not done:
        error_msg = "null"
        try:
            action = agent.get_action(obs)
            
            # Calculate metrics corresponding to step context to form sequence data for the grader
            heater_power = 5.0
            lights_power = 0.5
            energy = float((action.heater_control * heater_power) + (action.lights_control * lights_power))
            
            step_record = StepData(
                energy=energy,
                occupied=bool(obs.occupancy),
                temp=float(obs.indoor_temperature),
                heater_on=bool(action.heater_control),
                lights_on=bool(action.lights_control)
            )
            episode_data.append(step_record)
            
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            action_str = f"heater={action.heater_control},lights={action.lights_control}"
            
        except Exception as e:
            error_msg = str(e)
            action_str = "error"
            reward = 0.0
            done = True
            success = False

        print(f"[STEP] step={step_n} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
        sys.stdout.flush()
        step_n += 1

    try:
        final_score = evaluate_episode(task_name, episode_data)
    except Exception as e:
        success = False
        final_score = 0.0
        print(f"Error during final evaluation: {e}")
        sys.stdout.flush()

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step_n} score={final_score:.4f} rewards={rewards_str}")
    sys.stdout.flush()

if __name__ == "__main__":
    tasks_to_run = ["basic_thermostat", "day_night_control", "multiday_optimization"]
    for task in tasks_to_run:
        run_inference(task)
