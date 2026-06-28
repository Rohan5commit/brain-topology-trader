FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    ncps>=0.0.7 \
    yfinance>=0.2.36 \
    pyarrow>=14.0.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    requests>=2.31.0 \
    scikit-learn>=1.3.0

WORKDIR /workspace
