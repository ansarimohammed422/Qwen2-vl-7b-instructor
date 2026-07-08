# 1. Base Image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 2. Set Working Directory
WORKDIR /app

# 3. System & Python Updates
RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0 && \
    python -m pip install --upgrade pip

# 4. Install Dependencies
# We combine these into a single RUN command to keep the image layer size optimized.
RUN pip install --no-cache-dir \
    "torch>=2.4.0" \
    "transformers>=4.40.0" \
    "qwen-vl-utils" \
    "accelerate" \
    "pillow" \
    "torchvision" \
    "runpod"

# 5. Copy Handler
COPY handler.py .

# 6. Start the Handler
CMD ["python", "-u", "handler.py"]
