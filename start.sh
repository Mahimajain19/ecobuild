#!/bin/bash
# EcoBuild Startup Script

echo "Starting EcoBuild OpenEnv Server..."

# Start the FastAPI server to handle OpenEnv health checks/pings
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
