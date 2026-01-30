#!/usr/bin/env bash
set -e

# Ensure cwd is the repo root path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."


if [ -f ".env" ]; then
  set -a
  . ".env"
  set +a
fi

python models/standalone/wan26.py \
  --input_json data/dummy_input.json \
  --output_root ./results \
  --model wan2.6-i2v \
  --roll 1 \
  --resolution 720P


