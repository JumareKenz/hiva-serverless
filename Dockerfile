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

# Install vLLM 0.7.3 (stable for CUDA 12.4) and PyTorch
RUN pip install --no-cache-dir vllm==0.7.3 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Set HF token (passed from RunPod UI)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Create persistent cache directory
RUN mkdir -p /runpod-volume/model

# Expose port
EXPOSE 8000

# Run vLLM server (model downloads at runtime – no RUN timeout)
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", \
     "--quantization", "gptq", \
     "--dtype", "float16", \
     "--max-model-len", "32768", \
     "--gpu-memory-utilization", "0.92", \
     "--enable-chunked-prefill", \
     "--port", "8000", \
     "--host", "0.0.0.0"]
