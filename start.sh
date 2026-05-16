#!/usr/bin/env bash
# Start FastAPI, Gradio, and MLflow UI in the background.
# PIDs and logs land in .run/. Use `./stop.sh` to terminate cleanly.
#
# Flags:
#   --docker      use docker compose instead of local processes
#   --no-mlflow   skip the MLflow UI
#   --no-ui       skip the Gradio UI
#   --no-api      skip the FastAPI service

set -euo pipefail
cd "$(dirname "$0")"

USE_DOCKER=0
RUN_API=1
RUN_UI=1
RUN_MLFLOW=1
for arg in "$@"; do
  case "$arg" in
    --docker)    USE_DOCKER=1 ;;
    --no-api)    RUN_API=0 ;;
    --no-ui)     RUN_UI=0 ;;
    --no-mlflow) RUN_MLFLOW=0 ;;
    -h|--help)
      sed -n '1,12p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

if [[ $USE_DOCKER -eq 1 ]]; then
  echo "starting docker compose services..."
  exec docker compose -f deploy/docker-compose.yml up -d --build
fi

mkdir -p .run outputs

start() {
  local name="$1"; shift
  local cmd="$*"
  local pidfile=".run/${name}.pid"
  local logfile=".run/${name}.log"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "  $name already running (pid $(cat "$pidfile"))"
    return
  fi
  echo "  starting $name → $logfile"
  nohup bash -c "$cmd" > "$logfile" 2>&1 &
  echo $! > "$pidfile"
}

PROJECT_DIR="$(pwd)"
VENV="$PROJECT_DIR/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  if command -v uv >/dev/null; then
    echo "creating venv via uv sync..."
    uv sync
  else
    echo "error: .venv missing and uv not installed." >&2
    exit 1
  fi
fi

echo "starting local services (venv: $VENV)..."
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

# Use `python -m` to bypass any stale shebangs in .venv/bin/* console scripts.
PY="$VENV/bin/python"
# Run with CWD=src so `api.main:app` and `gradio_app` resolve as top-level imports.
SRC="$PROJECT_DIR/src"
[[ $RUN_API    -eq 1 ]] && start api    "cd '$SRC'         && '$PY' -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
[[ $RUN_UI     -eq 1 ]] && start ui     "cd '$SRC'         && '$PY' gradio_app.py"
[[ $RUN_MLFLOW -eq 1 ]] && start mlflow "cd '$PROJECT_DIR' && '$PY' -m mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db --default-artifact-root file://./outputs/mlruns --host 0.0.0.0 --port 5000"

sleep 1
echo ""
echo "running services:"
[[ $RUN_API    -eq 1 ]] && echo "  - FastAPI : http://localhost:8000  (docs: /docs)"
[[ $RUN_UI     -eq 1 ]] && echo "  - Gradio  : http://localhost:7860"
[[ $RUN_MLFLOW -eq 1 ]] && echo "  - MLflow  : http://localhost:5000"
echo ""
echo "stop with: ./stop.sh"
echo "logs:      tail -f .run/*.log"
