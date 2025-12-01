FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Switch to faster US mirrors and retry apt-get (fixes timeout errors)
RUN sed -i 's|http://archive.ubuntu.com|http://us.archive.ubuntu.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://us.archive.ubuntu.com|g' /etc/apt/sources.list && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install vLLM and dependencies
# Install PyTorch (CUDA 12.4)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Install vLLM
RUN pip install --no-cache-dir vllm==0.8.3


# Set HF token (replace with your token for gated models)
ARG HF_TOKEN
ENV HF_TOKEN=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

# Create persistent cache directory
RUN mkdir -p /runpod-volume/model

# Pre-download the model (optional but recommended for cold-start speed)
RUN python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
    local_dir="/runpod-volume/model",
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN"),
    resume_download=True
)
PY

# Expose port
EXPOSE 8000

# Run vLLM server
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/runpod-volume/model", \
     "--quantization", "gptq", \
     "--dtype", "float16", \
     "--max-model-len", "32768", \
     "--gpu-memory-utilization", "0.92", \
     "--enable-chunked-prefill", \
     "--port", "8000", \
     "--host", "0.0.0.0"]