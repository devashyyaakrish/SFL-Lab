#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -x .venv/bin/python || { echo 'Run bash scripts/setup.sh first.' >&2; exit 1; }
test -f frontend/dist/index.html || { echo 'Build the dashboard with npm --prefix frontend run build.' >&2; exit 1; }
exec .venv/bin/python -m sfl_lab serve --host 127.0.0.1 --port "${SFL_LAB_PORT:-8000}"
