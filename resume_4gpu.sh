#!/bin/bash
# Resume training from checkpoint with 4 GPUs and TensorBoard
# Usage:
#   ./resume_4gpu.sh latest                    # Resume from latest checkpoint
#   ./resume_4gpu.sh best                      # Resume from best checkpoint
#   ./resume_4gpu.sh <path>                    # Resume from specific checkpoint
#   ./resume_4gpu.sh latest 64                 # Resume with mini_batch_size=64

CHECKPOINT="${1:-latest}"
MINI_BATCH_SIZE="${2:-}"

echo "============================================================"
echo "Resuming 4-GPU training from checkpoint: $CHECKPOINT"
if [ -n "$MINI_BATCH_SIZE" ]; then
    echo "Mini batch size: $MINI_BATCH_SIZE"
fi
echo "GPUs: 4"
echo "TensorBoard: ENABLED"
echo "============================================================"

# Build command
CMD="src/train_improved.py --resume $CHECKPOINT --use_tensorboard"

# Add mini_batch_size if provided
if [ -n "$MINI_BATCH_SIZE" ]; then
    CMD="$CMD --mini_batch_size $MINI_BATCH_SIZE"
fi

# Launch with torchrun
torchrun --standalone --nproc_per_node=4 $CMD


