from fastapi import FastAPI
import uvicorn
from ecobuild_env.environment import EcoBuildEnv
from ecobuild_env.models import BuildingAction

app = FastAPI()
env = EcoBuildEnv()

@app.get("/")
def read_root():
    return {"status": "ok", "env": "ecobuild"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: BuildingAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
