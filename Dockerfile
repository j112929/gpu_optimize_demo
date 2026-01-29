# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# Copy requirements first to leverage cache if available (though we just install manually here)
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    fastapi \
    uvicorn \
    requests \
    sentencepiece \
    safetensors \
    streamlit \
    protobuf \
    pydantic

# Copy source code
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
# 8000: API Server
# 8501: Streamlit UI
EXPOSE 8000 8501

# Default command (Starts the API server)
# You can override this to run the UI: streamlit run examples/chat_ui.py
CMD ["python", "src/serving/server.py"]
