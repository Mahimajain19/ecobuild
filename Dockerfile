# Use a slim Python 3.10 image to minimize image size
FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
# Hugging Face Spaces run as a high-security non-root user (uid 1000)
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to /home/user/app
WORKDIR $HOME/app

# Copy dependency list and install them first for layer caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY --chown=user . .

# Ensure outputs directories exist and are writable
RUN mkdir -p outputs/logs outputs/evals && chown -R user:user outputs

# Expose Hugging Face Space default port
EXPOSE 7860

# Start the environment on port 7860
CMD ["sh", "start.sh"]
