# Multi-stage build: install deps once, then ship a single image with three
# entrypoints (api / ui / mlflow) selected via the CMD.
#
# Build (from project root):  docker build -t hand-gesture:latest -f deploy/Dockerfile .
# Run API:    docker run --rm -p 8000:8000 hand-gesture:latest api
# Run Gradio: docker run --rm -p 7860:7860 hand-gesture:latest ui
# Run MLflow: docker run --rm -p 5000:5000 hand-gesture:latest mlflow

FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv export --no-hashes --no-dev > requirements.txt \
    && pip install --no-cache-dir -r requirements.txt


FROM deps AS runtime

# Source code + the trained champion model files. The data CSV is not
# baked in (it's large and only needed for re-training, not for serving).
COPY src/ ./src/
COPY models/best_gesture_model.pkl models/label_encoder.pkl models/hand_landmarker.task ./models/

# CWD inside src/ matches how start.sh runs locally, so imports resolve
# the same way in dev and prod.
ENV PYTHONPATH=/app/src

EXPOSE 8000 7860 5000

COPY deploy/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["api"]
