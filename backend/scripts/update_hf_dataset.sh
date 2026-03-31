#!/bin/bash
# Rebuild HF dataset from fresh Wikimedia dump.
# Run via crontab: 0 4 1,15 * * /Users/leo/Code/mention-map/backend/scripts/update_hf_dataset.sh
#
# What it does:
#   1. Downloads the latest ru.wikisource dump (skips if unchanged)
#   2. Builds Parquet dataset from it
#   3. Pushes to HuggingFace Hub (requires HF_TOKEN in env or ~/.huggingface/token)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$BACKEND_DIR")"
OUTPUT_DIR="$PROJECT_DIR/hf_dataset"
LOG_FILE="$OUTPUT_DIR/build.log"

mkdir -p "$OUTPUT_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting dataset build..." | tee -a "$LOG_FILE"

cd "$BACKEND_DIR"

# Check if dump has been updated (compare file size with previous)
DUMP_FILE="$OUTPUT_DIR/ruwikisource-latest-pages-articles.xml.bz2"
OLD_SIZE=0
if [ -f "$DUMP_FILE" ]; then
    OLD_SIZE=$(stat -f%z "$DUMP_FILE" 2>/dev/null || stat -c%s "$DUMP_FILE" 2>/dev/null || echo 0)
fi

# Download new dump
uv run --extra hf python scripts/build_hf_from_dump.py \
    --output-dir "$OUTPUT_DIR" \
    --max-gb 100 \
    2>&1 | tee -a "$LOG_FILE"

# Check if dump changed
NEW_SIZE=0
if [ -f "$DUMP_FILE" ]; then
    NEW_SIZE=$(stat -f%z "$DUMP_FILE" 2>/dev/null || stat -c%s "$DUMP_FILE" 2>/dev/null || echo 0)
fi

if [ "$OLD_SIZE" = "$NEW_SIZE" ] && [ "$OLD_SIZE" != "0" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Dump unchanged ($NEW_SIZE bytes), skipping rebuild." | tee -a "$LOG_FILE"
    exit 0
fi

# Push to HF Hub
echo "$(date '+%Y-%m-%d %H:%M:%S') Pushing to HuggingFace Hub..." | tee -a "$LOG_FILE"
uv run --extra hf python scripts/build_hf_from_dump.py \
    --output-dir "$OUTPUT_DIR" \
    --push matyushkin/ru-wikisource-literature \
    2>&1 | tee -a "$LOG_FILE"

echo "$(date '+%Y-%m-%d %H:%M:%S') Done." | tee -a "$LOG_FILE"
