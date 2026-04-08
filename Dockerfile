# Use a slim Python 3.10 image to minimize image size
FROM python:3.10-slim

# Set working directory to /app
WORKDIR /app

# Copy dependency list and install them first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port (common for OpenEnv FastAPI servers)
EXPOSE 8000

# Start the environment
CMD ["sh", "start.sh"]
