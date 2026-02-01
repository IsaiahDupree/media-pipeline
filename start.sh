#!/bin/bash
# Start Media Pipeline Service
# Port: 6004

cd "$(dirname "$0")"

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set default port
export PORT=${PORT:-6004}

echo "🚀 Starting media-pipeline on port $PORT..."

# Check for required tools
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  Warning: ffmpeg not found - some features may not work"
fi

if ! command -v ffprobe &> /dev/null; then
    echo "⚠️  Warning: ffprobe not found - video analysis will fail"
fi

# Start the service
python app.py
