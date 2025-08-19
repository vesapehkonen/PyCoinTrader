#!/usr/bin/env bash
set -Eeuo pipefail

# cd to repo root (script can live anywhere in the repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # adjust if your script lives elsewhere

APPLY=false
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=true
fi

echo "Repository root: $(pwd)"
echo "Dry run: $([[ $APPLY == true ]] && echo 'NO (will delete)' || echo 'YES (preview only)')"
echo

# Patterns to clean
FILES_PATTERNS=(
  "*.pyc"
  "*.pyo"
  "*~"
  ".coverage"
)
DIR_PATTERNS=(
  "__pycache__"
  ".pytest_cache"
  ".mypy_cache"
  ".ipynb_checkpoints"
  "build"
  "dist"
  "*.egg-info"
)

# Files
for pat in "${FILES_PATTERNS[@]}"; do
  echo ">>> Matching files: $pat"
  if [[ $APPLY == true ]]; then
    find . -type f -name "$pat" -print -delete
  else
    find . -type f -name "$pat" -print
  fi
done

# Directories
for pat in "${DIR_PATTERNS[@]}"; do
  echo ">>> Matching directories: $pat"
  if [[ $APPLY == true ]]; then
    find . -type d -name "$pat" -prune -print -exec rm -rf {} +
  else
    find . -type d -name "$pat" -prune -print
  fi
done

echo
echo "Done. Re-run with --apply to actually delete."
