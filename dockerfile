# Use an official PyTorch CUDA image
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for image processing)
RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0

# Copy your requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler script
COPY rp_handler.py .

# Command to run your worker
CMD ["python", "-u", "rp_handler.py"]
