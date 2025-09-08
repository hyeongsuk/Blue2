#!/bin/bash
# Standalone EEG optimization runner
cd "/Users/hyeongsuk/Library/CloudStorage/OneDrive-개인/HS_논문작성/KOOS/code"

# Kill existing processes
pkill -f real_eeg_optimization.py

# Run with maximum independence
nohup python -u real_eeg_optimization.py > ../results/logs/eeg_optimization_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Started background process. Check logs in:"
echo "../results/logs/"
echo ""
echo "Monitor with: tail -f ../results/logs/eeg_optimization_*.log"
echo "Check status: ps aux | grep real_eeg_optimization"