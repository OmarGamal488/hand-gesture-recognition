#!/usr/bin/env bash
# Stop services started by ./start.sh.
#
# Flags:
#   --docker   stop docker compose stack instead
#   --clean    also remove .run/ logs and pidfiles

set -euo pipefail
cd "$(dirname "$0")"

USE_DOCKER=0
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --docker) USE_DOCKER=1 ;;
    --clean)  CLEAN=1 ;;
    -h|--help) sed -n '1,9p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

if [[ $USE_DOCKER -eq 1 ]]; then
  exec docker compose down
fi

if [[ ! -d .run ]]; then
  echo "nothing to stop (no .run/ directory)."
  exit 0
fi

for pidfile in .run/*.pid; do
  [[ -f "$pidfile" ]] || continue
  name=$(basename "$pidfile" .pid)
  pid=$(cat "$pidfile")
  if kill -0 "$pid" 2>/dev/null; then
    echo "  stopping $name (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "    forcing kill -9..."
      kill -9 "$pid" 2>/dev/null || true
    fi
  else
    echo "  $name not running (stale pid $pid)"
  fi
  rm -f "$pidfile"
done

if [[ $CLEAN -eq 1 ]]; then
  rm -rf .run/
  echo "removed .run/"
fi

echo "done."
