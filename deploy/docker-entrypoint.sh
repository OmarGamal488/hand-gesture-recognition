#!/usr/bin/env bash
set -euo pipefail

# CWD = /app so the model paths resolve via Path(__file__).parents[1] / "models"
# inside the image. The src/ subdir is on PYTHONPATH so imports work.
cmd="${1:-api}"
case "$cmd" in
  api)
    exec uvicorn api.main:app --host 0.0.0.0 --port 8000
    ;;
  ui)
    exec python /app/src/gradio_app.py
    ;;
  mlflow)
    mkdir -p /app/outputs
    exec mlflow ui --backend-store-uri sqlite:////app/outputs/mlflow.db \
                   --default-artifact-root file:///app/outputs/mlruns \
                   --host 0.0.0.0 --port 5000
    ;;
  *)
    exec "$@"
    ;;
esac
