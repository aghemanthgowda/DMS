# Container image for the Driver Monitoring System.
# Note: real camera/display access requires host device passthrough at runtime.
FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Best-effort model download at build time; skipped gracefully if offline.
RUN python scripts/download_model.py || true

CMD ["python", "main.py", "--no-audio"]
