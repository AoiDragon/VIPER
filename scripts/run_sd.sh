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

# Adapt the data into json
pyython data_helper.py 

python models/standalone/seedance.py \
  --input_json data/viper.json \
  --output_root ./results \
  --model doubao-seedance-1-5-pro-251215 \
  --roll 1

