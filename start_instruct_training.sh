#!/bin/bash
# Start instruction fine-tuning
# Usage:
#   ./start_instruct_training.sh                    # Single GPU
#   ./start_instruct_training.sh 4                  # 4 GPUs
#   ./start_instruct_training.sh 1 latest            # Resume from latest checkpoint
#   ./start_instruct_training.sh 4 step_001000       # Resume from step checkpoint (auto-prefixes checkpoints_instruct/)

NGPUS=${1:-1}
RESUME=${2:-}

# Auto-prefix checkpoints_instruct/ if resume is not "latest" and doesn't already have a path
if [ -n "$RESUME" ] && [ "$RESUME" != "latest" ]; then
    # If it doesn't start with checkpoints_instruct/ or a full path, add the prefix
    if [[ ! "$RESUME" =~ ^checkpoints_instruct/ ]] && [[ ! "$RESUME" =~ ^/ ]]; then
        RESUME="checkpoints_instruct/$RESUME"
        # Add .pt extension if not present
        if [[ ! "$RESUME" =~ \.pt$ ]]; then
            RESUME="${RESUME}.pt"
        fi
    fi
fi

echo "============================================================"
echo "Instruction Fine-Tuning for GPT-2"
echo "============================================================"
echo "Pre-trained model: checkpoints/best_model.pt"
echo "Dataset: Alpaca GPT-4 (52k samples)"
echo "Epochs: 5 (~4,000 steps)"
echo "GPUs: $NGPUS"
if [ -n "$RESUME" ]; then
    echo "Resume: $RESUME"
fi
echo "Checkpoint dir: checkpoints_instruct/"
echo "TensorBoard: runs_instruct/"
echo "============================================================"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Check if data is prepared
if [ ! -d "data_instruct" ]; then
    echo "⚠️  Instruction dataset not found!"
    echo "Please run: python src/prepare_instruct_dataset.py"
    echo ""
    exit 1
fi

# Start training
if [ "$NGPUS" -eq 1 ]; then
    # Single GPU
    if [ -n "$RESUME" ]; then
        python src/train_instruct.py \
            --pretrained_model checkpoints/best_model.pt \
            --max_steps 4000 \
            --learning_rate 2e-5 \
            --mini_batch_size 8 \
            --total_batch_size 65536 \
            --checkpoint_freq 100 \
            --eval_freq 100 \
            --use_tensorboard \
            --resume "$RESUME"
    else
        python src/train_instruct.py \
            --pretrained_model checkpoints/best_model.pt \
            --max_steps 4000 \
            --learning_rate 2e-5 \
            --mini_batch_size 8 \
            --total_batch_size 65536 \
            --checkpoint_freq 100 \
            --eval_freq 100 \
            --use_tensorboard
    fi
else
    # Multi-GPU with torchrun
    if [ -n "$RESUME" ]; then
        torchrun --standalone --nproc_per_node=$NGPUS src/train_instruct.py \
            --pretrained_model checkpoints/best_model.pt \
            --max_steps 4000 \
            --learning_rate 2e-5 \
            --mini_batch_size 8 \
            --total_batch_size 65536 \
            --checkpoint_freq 100 \
            --eval_freq 100 \
            --use_tensorboard \
            --resume "$RESUME"
    else
        torchrun --standalone --nproc_per_node=$NGPUS src/train_instruct.py \
            --pretrained_model checkpoints/best_model.pt \
            --max_steps 4000 \
            --learning_rate 2e-5 \
            --mini_batch_size 8 \
            --total_batch_size 65536 \
            --checkpoint_freq 100 \
            --eval_freq 100 \
            --use_tensorboard
    fi
fi

