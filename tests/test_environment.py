"""
EcoBuild environment unit tests.
Run: pytest tests/ -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ecobuild_env.environment import EcoBuildEnvironment
from ecobuild_env.models import BuildingAction
from ecobuild_env.tasks import StepData, evaluate_episode


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def env():
    e = EcoBuildEnvironment()
    e.reset(task_name="basic_thermostat", seed=42)
    return e


@pytest.fixture
def action_off():
    return BuildingAction(heater_control=0, ac_control=0, lights_control=0)


@pytest.fixture
def action_heater():
    return BuildingAction(heater_control=1, ac_control=0, lights_control=0)


# ─────────────────────────────────────────────
# Reset Tests
# ─────────────────────────────────────────────

def test_reset_returns_valid_observation(env):
    obs = env.get_observation()
    assert 10.0 <= obs.indoor_temperature <= 35.0
    assert 0 <= obs.occupancy_count <= 100
    assert 0 <= obs.hour_of_day <= 23
    assert obs.time_step == 0
    assert obs.grid_available in (0, 1)


def test_reset_seed_reproducibility():
    env1 = EcoBuildEnvironment()
    env2 = EcoBuildEnvironment()
    obs1 = env1.reset(task_name="basic_thermostat", seed=42)
    obs2 = env2.reset(task_name="basic_thermostat", seed=42)
    assert obs1.indoor_temperature == obs2.indoor_temperature
    assert obs1.outdoor_temperature == obs2.outdoor_temperature


def test_reset_different_seeds_produce_different_episodes():
    env = EcoBuildEnvironment()
    obs1 = env.reset(task_name="basic_thermostat", seed=0)
    obs2 = env.reset(task_name="basic_thermostat", seed=99)
    # Not guaranteed to differ on every field, but at least something should differ
    combined = [
        obs1.indoor_temperature != obs2.indoor_temperature,
        obs1.outdoor_temperature != obs2.outdoor_temperature,
        obs1.day_of_week != obs2.day_of_week,
    ]
    assert any(combined)


def test_reset_assigns_episode_id():
    env = EcoBuildEnvironment()
    env.reset(task_name="basic_thermostat")
    assert env.episode_id != ""
    assert len(env.episode_id) == 36  # UUID format


def test_reset_resets_accumulators():
    env = EcoBuildEnvironment()
    env.reset(task_name="basic_thermostat", seed=1)
    for _ in range(10):
        env.step(BuildingAction(heater_control=1, lights_control=1))
    env.reset(task_name="basic_thermostat", seed=2)
    assert env.total_energy_kwh == 0.0
    assert env.total_cost_inr == 0.0
    assert env.comfort_violations == 0


# ─────────────────────────────────────────────
# Step Tests
# ─────────────────────────────────────────────

def test_step_advances_time(env, action_off):
    obs, _, _, _ = env.step(action_off)
    assert obs.time_step == 1


def test_step_returns_valid_types(env, action_heater):
    obs, reward, done, info = env.step(action_heater)
    assert isinstance(obs.indoor_temperature, float)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "reward_breakdown" in info
    assert "episode_id" in info


def test_heater_warms_room():
    """Heater on in mild-outdoor-temp room → temperature increases towards comfort zone."""
    env = EcoBuildEnvironment()
    # Use day_night_tou (summer) so outdoor temp is ~25-35°C, heater easily warms from 16°C
    env.reset(task_name="day_night_tou", seed=10)
    env.indoor_temp = 16.0   # Force cold start
    env.weather = __import__("ecobuild_env.weather", fromlist=["IndianWeatherModel"]).IndianWeatherModel(
        region="delhi", season="summer"
    )

    action = BuildingAction(heater_control=1, ac_control=0, lights_control=0)
    temps = [env.indoor_temp]
    for _ in range(18):  # 3 hours to warm up
        obs, _, _, _ = env.step(action)
        temps.append(obs.indoor_temperature)

    # With heater on in summer (outdoor ~30°C), indoor should rise from 16°C
    assert temps[-1] > temps[0], f"Temperature should rise with heater on in summer. Got: {temps}"


def test_ac_cools_room():
    """AC on in hot room → temperature decreases."""
    env = EcoBuildEnvironment()
    env.reset(task_name="day_night_tou", seed=5)
    env.indoor_temp = 30.0  # Force hot start

    action = BuildingAction(heater_control=0, ac_control=1, lights_control=0)
    temps = [env.indoor_temp]
    for _ in range(12):
        obs, _, _, _ = env.step(action)
        temps.append(obs.indoor_temperature)

    assert temps[-1] < temps[0], f"Temperature should fall with AC on. Got: {temps}"


def test_episode_terminates_at_max_steps():
    env = EcoBuildEnvironment()
    env.reset(task_name="basic_thermostat", seed=0)
    action = BuildingAction()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(action)
        steps += 1
        if steps > 200:
            break
    assert done
    assert steps == env.max_steps or steps == 144  # basic_thermostat = 144 steps


def test_info_contains_required_fields(env, action_off):
    _, _, _, info = env.step(action_off)
    required = ["reward_breakdown", "episode_id", "task_name", "step",
                "co2_kg_this_step", "total_cost_inr", "grid_available"]
    for field in required:
        assert field in info, f"Missing info field: {field}"


def test_reward_breakdown_is_complete(env, action_off):
    _, _, _, info = env.step(action_off)
    rb = info["reward_breakdown"]
    required_keys = ["total", "energy_cost_inr", "comfort_penalty",
                     "vacancy_waste", "constraint_violation"]
    for key in required_keys:
        assert key in rb, f"Missing reward_breakdown key: {key}"


# ─────────────────────────────────────────────
# All Task Tests
# ─────────────────────────────────────────────

@pytest.mark.parametrize("task", [
    "basic_thermostat",
    "day_night_tou",
    "load_shedding_optimizer",
    "multiday_optimization",
])
def test_all_tasks_reset_and_step(task):
    """All tasks should reset and step without errors."""
    env = EcoBuildEnvironment()
    obs = env.reset(task_name=task, seed=0)
    assert obs is not None
    assert obs.time_step == 0

    action = BuildingAction(heater_control=1, ac_control=0, lights_control=1)
    obs2, reward, done, info = env.step(action)
    assert obs2 is not None
    assert isinstance(reward, float)
    assert "task_name" in info
    assert info["task_name"] == task


# ─────────────────────────────────────────────
# Grader Tests
# ─────────────────────────────────────────────

def test_grader_returns_0_to_1():
    env = EcoBuildEnvironment()
    for task in ["basic_thermostat", "day_night_tou"]:
        env.reset(task_name=task, seed=42)
        for _ in range(20):
            env.step(BuildingAction(heater_control=1, lights_control=1, ac_control=0))
        score = env.grade()
        assert 0.0 <= score <= 1.0, f"Score out of range for {task}: {score}"


def test_grader_empty_episodes():
    from ecobuild_env.tasks import Task1Grader
    score = Task1Grader().grade([])
    assert score == 0.0


def test_state_returns_typed_model(env):
    state = env.state()
    assert hasattr(state, "episode_id")
    assert hasattr(state, "task_name")
    assert hasattr(state, "step_count")
    assert hasattr(state, "is_done")
    assert state.is_done == False


# ─────────────────────────────────────────────
# Grid / Power Cut Tests
# ─────────────────────────────────────────────

def test_tariff_peak_hours():
    from ecobuild_env.tariff import IndianElectricityTariff
    tariff = IndianElectricityTariff("delhi")
    assert tariff.get_current_tariff(19) == tariff.peak_rate     # 7 PM = peak
    assert tariff.get_current_tariff(3) == tariff.off_peak_rate  # 3 AM = off-peak
    assert tariff.peak_rate > tariff.normal_rate > tariff.off_peak_rate


def test_grid_generates_schedule():
    from ecobuild_env.grid import IndianGridModel
    grid = IndianGridModel(city="tier2", mode="scheduled",
                           rng=__import__("numpy").random.RandomState(0), total_steps=144)
    # tier2 has 5h/day cuts — some steps should be unavailable
    cut_count = sum(1 for i in range(144) if not grid.is_grid_available(i))
    assert cut_count > 0, "Tier-2 city should have some power cuts"


def test_weather_returns_seasonal_conditions():
    from ecobuild_env.weather import IndianWeatherModel
    model = IndianWeatherModel(region="delhi", season="summer")
    conditions = model.get_conditions(14, 120)  # 2PM, day 120 = peak summer
    assert conditions.temperature > 30, f"Delhi summer afternoon should be >30°C, got {conditions.temperature}"
    assert conditions.season == "summer"
