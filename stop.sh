#!/bin/bash
# Stop Media Pipeline Service

echo "🛑 Stopping media-pipeline..."

# Find and kill the process
pkill -f "python app.py" 2>/dev/null || true
pkill -f "media-pipeline" 2>/dev/null || true

# Kill by port if still running
lsof -ti:6004 | xargs kill -9 2>/dev/null || true

echo "✅ media-pipeline stopped"
