#!/bin/bash
# Start instruction fine-tuning
# Usage:
#   ./start_instruct_training.sh              # Single GPU
#   ./start_instruct_training.sh 4            # 4 GPUs

NGPUS=${1:-1}

echo "============================================================"
echo "Instruction Fine-Tuning for GPT-2"
echo "============================================================"
echo "Pre-trained model: checkpoints/best_model.pt"
echo "Dataset: Alpaca GPT-4 (52k samples)"
echo "GPUs: $NGPUS"
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
    python src/train_instruct.py \
        --pretrained_model checkpoints/best_model.pt \
        --max_steps 2000 \
        --learning_rate 2e-5 \
        --mini_batch_size 8 \
        --total_batch_size 65536 \
        --checkpoint_freq 100 \
        --eval_freq 100 \
        --use_tensorboard
else
    # Multi-GPU with torchrun
    torchrun --standalone --nproc_per_node=$NGPUS src/train_instruct.py \
        --pretrained_model checkpoints/best_model.pt \
        --max_steps 2000 \
        --learning_rate 2e-5 \
        --mini_batch_size 8 \
        --total_batch_size 65536 \
        --checkpoint_freq 100 \
        --eval_freq 100 \
        --use_tensorboard
fi

