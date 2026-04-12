"""
EcoBuild Environment Client.
Implements the OpenEnv EnvClient interface for connecting to a deployed EcoBuild server.
Supports both async and sync usage patterns.
"""

from __future__ import annotations
from typing import Optional, Any

try:
    from openenv.core.client import EnvClient
    _HAS_OPENENV = True
except ImportError:
    _HAS_OPENENV = False

from .models import BuildingAction, BuildingObservation, EcoBuildState


if _HAS_OPENENV:
    class EcoBuildEnv(EnvClient):
        """
        Remote client for the EcoBuild building energy management environment.

        Usage (async):
            env = EcoBuildEnv(base_url="https://your-hf-space.hf.space")
            obs = await env.reset(task_name="basic_thermostat", seed=42)
            obs, reward, done, info = await env.step(BuildingAction(heater_control=1))

        Usage (sync):
            env = EcoBuildEnv.sync(base_url="https://your-hf-space.hf.space")
            obs = env.reset(task_name="day_night_tou", seed=0)
        """
        action_type = BuildingAction
        observation_type = BuildingObservation

        async def reset(
            self,
            task_name: str = "basic_thermostat",
            seed: Optional[int] = None,
        ) -> BuildingObservation:
            """Reset environment to a new episode."""
            return await self._reset({"task_name": task_name, "seed": seed})

        async def step(self, action: BuildingAction):
            """Perform one step in the environment."""
            return await self._step(action)

        async def state(self) -> EcoBuildState:
            """Retrieve current episode state."""
            return await self._state()

        async def grade(self) -> float:
            """Get the current episode score [0.0, 1.0]."""
            result = await self._get("/grade")
            return result.get("score", 0.0)

        async def list_tasks(self) -> list:
            """List all available tasks."""
            result = await self._get("/tasks")
            return result.get("tasks", [])

else:
    # Fallback when openenv-core is not installed — lightweight websocket client
    import json
    import asyncio

    class EcoBuildEnv:
        """
        Lightweight EcoBuild client (no openenv-core dependency).
        Connects directly via WebSocket.

        Usage:
            env = EcoBuildEnv(base_url="http://localhost:8000")
            # Use with asyncio or call .connect() first
        """

        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url.rstrip("/")
            self._ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
            self._ws = None

        async def connect(self):
            """Establish WebSocket connection."""
            try:
                import websockets
                self._ws = await websockets.connect(self._ws_url)
            except ImportError:
                raise ImportError("Install 'websockets' to use EcoBuildEnv without openenv-core: pip install websockets")

        async def reset(self, task_name: str = "basic_thermostat", seed: Optional[int] = None) -> dict:
            """Reset the environment via WebSocket."""
            if self._ws is None:
                await self.connect()
            await self._ws.send(json.dumps({"type": "reset", "task_name": task_name, "seed": seed}))
            response = json.loads(await self._ws.recv())
            return response.get("observation", {})

        async def step(self, action: dict) -> tuple:
            """Perform a step via WebSocket."""
            if self._ws is None:
                await self.connect()
            await self._ws.send(json.dumps({"type": "step", "action": action}))
            response = json.loads(await self._ws.recv())
            return (
                response.get("observation", {}),
                response.get("reward", 0.0),
                response.get("done", False),
                response.get("info", {}),
            )

        async def grade(self) -> float:
            if self._ws is None:
                await self.connect()
            await self._ws.send(json.dumps({"type": "grade"}))
            response = json.loads(await self._ws.recv())
            return response.get("score", 0.0)

        async def close(self):
            if self._ws:
                await self._ws.close()
                self._ws = None

        def __repr__(self):
            return f"EcoBuildEnv(base_url='{self.base_url}')"
