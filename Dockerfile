# FROM --platform=linux/amd64 python:3.9-slim

# WORKDIR /app

# COPY requirements.txt .
# # Install CPU-only torch first to ensure a smaller image
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# RUN pip install --no-cache-dir -r requirements.txt

# COPY ./src /app/src
# COPY ./models /app/models

# ENTRYPOINT ["python", "src/run_inference.py"]



# Use a specific, lightweight base image compatible with AMD64
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies, including the CPU-only version of PyTorch
# The --no-install-recommends is for Debian-based images to reduce bloat
RUN apt-get update && apt-get install -y --no-install-recommends \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy your application code and the trained model assets
COPY ./src /app/src
COPY ./models /app/models

# Specify the command to run on container start
ENTRYPOINT ["python", "-m", "src.run_final"]