#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"

find "$ROOT" -type f -name '*.py' \
  -not -path '*/.git/*' \
  -not -path '*/.venv/*' \
  -not -path '*/venv/*' \
  -not -path '*/__pycache__/*' \
  -print0 \
    | xargs -0 awk 'NF { count++ } END { print count+0 }'
