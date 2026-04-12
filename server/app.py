"""
EcoBuild FastAPI Server.
Exposes HTTP endpoints + WebSocket for the OpenEnv standard API.
Supports concurrent sessions via per-connection EcoBuildEnvironment instances.
"""

import time
import uuid
import logging
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ecobuild_env.environment import EcoBuildEnvironment
from ecobuild_env.models import BuildingAction
from ecobuild_env.task_configs import list_tasks

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] ecobuild.server | %(message)s",
)
logger = logging.getLogger("ecobuild.server")

START_TIME = time.time()

app = FastAPI(
    title="EcoBuild Environment Server",
    description="OpenEnv-compliant smart building energy management RL environment",
    version="1.0.0",
)

# Shared HTTP env (for simple REST usage)
_http_env = EcoBuildEnvironment()

# WebSocket session registry: session_id → EcoBuildEnvironment
sessions: dict = {}


# ─────────────────────────────────────────────
# HTTP Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def read_root():
    return {
        "service": "ecobuild",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
    }


@app.get("/health")
def health():
    """OpenEnv platform health check endpoint."""
    return {
        "status": "healthy",
        "service": "ecobuild",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - START_TIME),
        "active_connections": len(sessions),
        "available_tasks": [t["name"] for t in list_tasks()],
        "features": [
            "thermal_mass", "tou_tariffs", "load_shedding",
            "solar_battery", "aqi_modeling", "festival_calendar",
            "genset", "seed_control", "episode_tracking",
        ],
    }


@app.get("/tasks")
def get_tasks():
    """List all available tasks with metadata."""
    return {"tasks": list_tasks()}


@app.post("/reset")
def reset(task_name: str = "basic_thermostat", seed: Optional[int] = None):
    """Reset the HTTP env to a new episode."""
    obs = _http_env.reset(task_name=task_name, seed=seed)
    logger.info(f"HTTP reset | task={task_name} seed={seed} episode={_http_env.episode_id[:8]}")
    return {
        "observation": obs.model_dump(),
        "episode_id": _http_env.episode_id,
        "task_name": task_name,
        "max_steps": _http_env.max_steps,
    }


@app.post("/step")
def step(action: BuildingAction):
    """Perform one environment step."""
    obs, reward, done, info = _http_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }


@app.get("/state")
def state():
    """Return current episode state."""
    return _http_env.state().model_dump()


@app.get("/grade")
def grade():
    """Return the current episode score [0.0, 1.0]."""
    score = _http_env.grade()
    from ecobuild_env.tasks import evaluate_episode_breakdown
    breakdown = {}
    if _http_env.episode_data:
        try:
            breakdown = evaluate_episode_breakdown(_http_env.task_name, _http_env.episode_data)
        except Exception:
            breakdown = {"total_score": score}
    return {
        "score": score,
        "breakdown": breakdown,
        "episode_id": _http_env.episode_id,
        "task_name": _http_env.task_name,
        "steps_completed": _http_env.current_step,
    }


# ─────────────────────────────────────────────
# WebSocket Endpoint
# ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket session for OpenEnv EnvClient.

    Protocol:
      Client → {"type": "reset", "task_name": "...", "seed": 42}
      Server ← {"type": "reset", "observation": {...}, "episode_id": "..."}

      Client → {"type": "step", "action": {"heater_control": 1, ...}}
      Server ← {"type": "step", "observation": {...}, "reward": 0.0, "done": false, "info": {...}}

      Client → {"type": "state"}
      Server ← {"type": "state", "state": {...}}

      Client → {"type": "grade"}
      Server ← {"type": "grade", "score": 0.72}
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    env = EcoBuildEnvironment()
    sessions[session_id] = env
    logger.info(f"WS connect | session={session_id[:8]} | active={len(sessions)}")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "reset":
                task_name = data.get("task_name", "basic_thermostat")
                seed = data.get("seed", None)
                obs = env.reset(task_name=task_name, seed=seed)
                logger.info(f"WS reset | session={session_id[:8]} task={task_name} seed={seed}")
                await websocket.send_json({
                    "type": "reset",
                    "observation": obs.model_dump(),
                    "episode_id": env.episode_id,
                    "task_name": task_name,
                    "max_steps": env.max_steps,
                })

            elif msg_type == "step":
                action_data = data.get("action", {})
                try:
                    action = BuildingAction(**action_data)
                    obs, reward, done, info = env.step(action)
                    await websocket.send_json({
                        "type": "step",
                        "observation": obs.model_dump(),
                        "reward": float(reward),
                        "done": bool(done),
                        "info": info,
                    })
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type == "state":
                await websocket.send_json({
                    "type": "state",
                    "state": env.state().model_dump(),
                })

            elif msg_type == "grade":
                score = env.grade()
                await websocket.send_json({
                    "type": "grade",
                    "score": score,
                    "episode_id": env.episode_id,
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: '{msg_type}'. Use reset|step|state|grade.",
                })

    except WebSocketDisconnect:
        logger.info(f"WS disconnect | session={session_id[:8]}")
    except Exception as e:
        logger.error(f"WS error | session={session_id[:8]} error={e}")
    finally:
        sessions.pop(session_id, None)
        logger.info(f"WS cleanup | session={session_id[:8]} | remaining={len(sessions)}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    logger.info("Starting EcoBuild server on 0.0.0.0:8000")
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, log_level=LOG_LEVEL.lower())


if __name__ == "__main__":
    main()
