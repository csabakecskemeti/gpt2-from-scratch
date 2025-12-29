#!/bin/bash
# Start interactive chat with fine-tuned model
# Usage:
#   ./chat.sh                              # Use latest checkpoint
#   ./chat.sh checkpoints_instruct/step_001000.pt  # Use specific checkpoint

CHECKPOINT=${1:-checkpoints_instruct/latest.pt}

echo "============================================================"
echo "Interactive Chat with Instruction-Tuned GPT-2"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "============================================================"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "⚠️  Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -lh checkpoints_instruct/*.pt 2>/dev/null || echo "  (none found)"
    echo ""
    exit 1
fi

# Start chat
python src/chat.py --checkpoint "$CHECKPOINT"

