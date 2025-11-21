# Option 1: Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy the Python script into container
COPY compute_bench.py /app/

# Set the default command to run the script
CMD ["python", "compute_bench.py"]
