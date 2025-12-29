#!/bin/bash
# Resume training from checkpoint with TensorBoard
# Usage:
#   ./resume_training.sh latest                    # Resume from latest checkpoint
#   ./resume_training.sh best                      # Resume from best checkpoint
#   ./resume_training.sh <path>                    # Resume from specific checkpoint
#   ./resume_training.sh latest 64                 # Resume with mini_batch_size=64
#   ./resume_training.sh checkpoints/rolling_step_014000.pt 64  # Specific checkpoint with batch size

CHECKPOINT="${1:-latest}"
MINI_BATCH_SIZE="${2:-}"

echo "============================================================"
echo "Resuming training from checkpoint: $CHECKPOINT"
if [ -n "$MINI_BATCH_SIZE" ]; then
    echo "Mini batch size: $MINI_BATCH_SIZE"
fi
echo "TensorBoard: ENABLED"
echo "============================================================"

# Build command
CMD="python src/train_improved.py --resume $CHECKPOINT --use_tensorboard"

# Add mini_batch_size if provided
if [ -n "$MINI_BATCH_SIZE" ]; then
    CMD="$CMD --mini_batch_size $MINI_BATCH_SIZE"
fi

# Check if running with DDP
if [ -z "$RANK" ]; then
    # Single GPU or should be launched via torchrun externally
    eval $CMD
else
    # Multi-GPU (launched via torchrun)
    eval $CMD
fi

