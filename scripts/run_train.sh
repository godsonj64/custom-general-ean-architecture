#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

echo "Starting training with config: ${CONFIG}"
python -m src.train --config "${CONFIG}"
