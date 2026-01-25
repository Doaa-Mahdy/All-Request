FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Update package lists and install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3.11 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest
RUN pip install --upgrade pip

# Copy requirements.txt
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install -q -r requirements.txt

# Copy all project files
COPY . /app/

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV USE_LLM=1

# Default command
CMD ["python3", "handler.py"]
