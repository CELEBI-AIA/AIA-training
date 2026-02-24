#!/bin/bash
echo "🚨 FORCE STOPPING TRAINING & CLEANING MEMORY 🚨"

# Kill specific training processes
echo "Killing train.py processes..."
pkill -9 -f "uav_training/train.py"

# Kill YOLO related processes
echo "Killing YOLO processes..."
pkill -9 -f "yolo"

# Kill build_dataset if running
echo "Killing dataset builders..."
pkill -9 -f "build_dataset.py"

# General python kill if specific ones fail (Optional, commented out to be safe)
# pkill -9 python3

echo "✅ Processes killed."

# Check GPU status
if command -v nvidia-smi &> /dev/null
then
    echo "📊 Current GPU Status:"
    nvidia-smi
else
    echo "⚠️ nvidia-smi not found, cannot check GPU status."
fi

echo "🧹 Cleanup Complete. RAM and VRAM should be released."
