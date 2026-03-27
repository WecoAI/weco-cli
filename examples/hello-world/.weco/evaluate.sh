#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Run evaluation and transform output to remove "x" suffix from speedup
uv run python evaluate.py --path .weco/optimize.py 2>&1 | sed 's/speedup: \([0-9.]*\)x/speedup: \1/'
