#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
setup_cache="${SFL_LAB_CACHE_DIR:-$PWD/.cache}"
mkdir -p "$setup_cache/pip" "$setup_cache/npm"
python3 -m venv .venv
.venv/bin/python -m pip install --cache-dir "$setup_cache/pip" -r requirements.lock
.venv/bin/python -m pip install --cache-dir "$setup_cache/pip" --no-build-isolation -e .
npm --prefix frontend ci --cache "$setup_cache/npm"
npm --prefix frontend run build
.venv/bin/python -m pip --cache-dir "$setup_cache/pip" check
