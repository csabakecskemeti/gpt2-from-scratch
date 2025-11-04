#!/bin/bash
# Resume training from checkpoint
# Usage:
#   ./resume_training.sh latest           # Resume from latest checkpoint
#   ./resume_training.sh best             # Resume from best checkpoint
#   ./resume_training.sh <path>           # Resume from specific checkpoint

CHECKPOINT="${1:-latest}"

echo "============================================================"
echo "Resuming training from checkpoint: $CHECKPOINT"
echo "============================================================"

# Check if running with DDP
if [ -z "$RANK" ]; then
    # Single GPU
    python src/train_improved.py --resume "$CHECKPOINT"
else
    # Multi-GPU (should be launched via torchrun)
    python src/train_improved.py --resume "$CHECKPOINT"
fi

