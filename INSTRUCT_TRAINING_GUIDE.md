# 🎯 Instruction Fine-Tuning Quick Start Guide

## Overview

This guide walks you through fine-tuning your pre-trained GPT-2 model on instruction-response pairs to create a chat assistant.

---

## 📋 Prerequisites

✅ Pre-trained GPT-2 model at `checkpoints/best_model.pt`
✅ Virtual environment activated (`.venv`)
✅ ~20GB free disk space for instruction dataset
✅ GPU with sufficient memory (tested on 120GB Blackwell)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Instruction Dataset

Download and prepare the Alpaca GPT-4 dataset (52k instruction-response pairs):

```bash
cd /home/kecso/Documents/workspace/training/gpt-2/gpt2-from-scratch
source .venv/bin/activate

python src/prepare_instruct_dataset.py
```

**Expected output:**
```
Downloading dataset from Hugging Face...
Total examples: 52,002
Train examples: 49,401
Val examples: 2,601

Processing examples...
✓ Dataset preparation complete!
Train shards: XXX
Val shards: XX
Output directory: data_instruct/
```

**Time**: ~5-10 minutes
**Output**: `data_instruct/train/` and `data_instruct/val/` folders with tokenized shards

---

### Step 2: Start Fine-Tuning

```bash
# Single GPU (8-12 hours)
./start_instruct_training.sh

# OR 4 GPUs (2-4 hours)
./start_instruct_training.sh 4
```

**What it does:**
- Loads your pre-trained model from `checkpoints/best_model.pt`
- Fine-tunes on instruction data with lower learning rate (2e-5)
- Saves checkpoints to `checkpoints_instruct/`
- Logs to TensorBoard at `runs_instruct/`

**Expected output:**
```
Loading pre-trained model from checkpoints/best_model.pt...
✓ Loaded pre-trained model
  Model parameters: 124,439,808

Starting Instruction Fine-Tuning
step     0 | loss 3.2145 | lr 2.00e-07 | norm 0.8234 | dt 245ms | tok/sec 524288
step    10 | loss 2.8421 | lr 2.00e-06 | norm 0.7123 | dt 243ms | tok/sec 528394
...
```

**Monitor with TensorBoard:**
```bash
tensorboard --logdir=runs_instruct/ --port=6007
```

---

### Step 3: Chat with Your Model

Once training is complete (or at any checkpoint):

```bash
# Use latest checkpoint
./chat.sh

# OR use specific checkpoint
./chat.sh checkpoints_instruct/step_001000.pt
```

**Example conversation:**
```
You: What is machine learning?