#!/bin/bash
# Convert instruction fine-tuned checkpoint to Hugging Face format
# Usage:
#   ./convert_to_hf.sh                              # Convert latest checkpoint
#   ./convert_to_hf.sh step_002000                  # Convert specific step
#   ./convert_to_hf.sh checkpoints_instruct/latest.pt my_model  # Custom paths

CHECKPOINT=${1:-checkpoints_instruct/latest.pt}
OUTPUT_DIR=${2:-hf_model}

# Auto-prefix checkpoints_instruct/ if needed
if [[ ! "$CHECKPOINT" =~ ^checkpoints_instruct/ ]] && [[ ! "$CHECKPOINT" =~ ^/ ]]; then
    if [[ "$CHECKPOINT" == "latest" ]]; then
        CHECKPOINT="checkpoints_instruct/latest.pt"
    else
        CHECKPOINT="checkpoints_instruct/$CHECKPOINT"
        # Add .pt extension if not present
        if [[ ! "$CHECKPOINT" =~ \.pt$ ]]; then
            CHECKPOINT="${CHECKPOINT}.pt"
        fi
    fi
fi

echo "============================================================"
echo "Converting Checkpoint to Hugging Face Format"
echo "============================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
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

# Run conversion (using bfloat16 by default)
python src/convert_to_safetensor.py "$CHECKPOINT" --output_dir "$OUTPUT_DIR" --dtype bfloat16

echo ""
echo "✓ Conversion complete! Model saved to: $OUTPUT_DIR"
echo ""
echo "To use with Hugging Face:"
echo "  from transformers import GPT2LMHeadModel"
echo "  model = GPT2LMHeadModel.from_pretrained('$OUTPUT_DIR')"
echo ""


