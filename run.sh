#!/usr/bin/env bash
set -euo pipefail

# Forward optional CLI overrides, for example:
# bash run.sh --model logistic_regression --output-dir artifacts/logistic
python -m src.train --config config/default.yaml "$@"
