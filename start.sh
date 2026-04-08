#!/bin/bash
# EcoBuild Startup Script

echo "Starting EcoBuild Inference and Server..."

# Run inference in the background to avoid blocking the server startup
# and to capture logs in real-time.
python inference.py &

# Start the FastAPI server to handle OpenEnv health checks/pings
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
