FROM python:3.13-slim

# System deps for OpenCV, MediaPipe, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (saves ~1.5 GB vs CUDA build).
# Must come before requirements.txt so transitive deps don't pull CUDA torch.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY src/ src/
COPY pages/ pages/
COPY scripts/ scripts/

# Copy artifacts (models + metadata + face_landmarker)
COPY artifacts/ artifacts/

# Copy dataset (needed for Dataset Explorer and Feature Engineering pages)
COPY data/ data/

# Streamlit config
RUN mkdir -p .streamlit
COPY .streamlit/config.toml .streamlit/config.toml

# Cloud Run sets PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port", "8080", \
    "--server.address", "0.0.0.0", \
    "--server.headless", "true", \
    "--browser.gatherUsageStats", "false"]
