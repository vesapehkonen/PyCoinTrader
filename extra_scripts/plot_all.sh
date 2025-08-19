#!/bin/bash

# Exit on error
set -e

# Check if directory path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

SEARCH_DIR="$1"

# Find all data.pkl files and loop over them
find "$SEARCH_DIR" -type f -name "data.pkl" | while read -r file; do
    echo "Processing: $file"
    python plot_performance.py "$file"
done
