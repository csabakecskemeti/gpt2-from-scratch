#!/bin/bash
# List all available checkpoints

echo "============================================================"
echo "Available Checkpoints"
echo "============================================================"

python src/train_improved.py --list_checkpoints

