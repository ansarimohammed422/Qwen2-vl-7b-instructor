# Use an official PyTorch CUDA image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for image processing)
RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0

# Copy your requirements and install Python packages
COPY requirements.txt .
RUN pip install -U --no-cache-dir -r requirements.txt

# Copy your handler script
# Copy your handler script
COPY handler.py .
# Command to run your worker
CMD ["python", "-u", "handler.py"]
