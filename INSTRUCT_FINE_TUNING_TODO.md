# 🎯 Instruction Fine-Tuning Implementation TODO

## 📋 Implementation Checklist

### Phase 1: Data Preparation ✅ COMPLETE
- [x] Create `src/prepare_instruct_dataset.py`
  - Download Alpaca GPT-4 dataset (52k samples)
  - Format with instruction template
  - Tokenize and save as shards
  - Create train/val split (95/5)
  
### Phase 2: Dataloader ✅ COMPLETE
- [x] Create `src/dataloader_instruct.py`
  - Simple dataloader for instruction shards
  - Similar to existing `dataloader.py`
  - Handles instruction format

### Phase 3: Training Script ✅ COMPLETE
- [x] Create `src/train_instruct.py`
  - Copy infrastructure from `train_improved.py`
  - Load pre-trained model from `checkpoints/best_model.pt`
  - Fine-tune with lower learning rate (2e-5)
  - Save to `checkpoints_instruct/` folder
  - TensorBoard logging to `runs_instruct/`
  - All existing features: checkpointing, resume, DDP support
  - **Fixed**: Batch size calculation (65536 tokens total)

### Phase 4: Inference ✅ COMPLETE
- [x] Create `src/chat.py`
  - Interactive chat interface
  - Handles instruction template automatically
  - Clean user experience

### Phase 5: Documentation & Scripts ✅ COMPLETE
- [x] Create `start_instruct_training.sh`
  - Helper script to start fine-tuning
  
- [x] Create `chat.sh`
  - Helper script to launch chat interface

- [x] Create `INSTRUCT_TRAINING_GUIDE.md`
  - Quick start guide
  - Usage examples
  - Troubleshooting

---

## 🎯 Key Parameters

### Dataset
- **Name**: `vicgalle/alpaca-gpt4`
- **Size**: 52,000 instruction-response pairs
- **Split**: 95% train (49,400), 5% val (2,600)

### Template Format
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

### Training Hyperparameters
```python
learning_rate = 2e-5              # 30x lower than pre-training
max_steps = 2000                  # ~3 epochs for 52k samples
warmup_steps = 100                # 5% warmup
mini_batch_size = 8               # Smaller batches
total_batch_size = 128            # Effective batch size
gradient_accumulation = 16        # 128 / 8 = 16
eval_freq = 100                   # Frequent evaluation
checkpoint_freq = 100             # Frequent checkpoints
weight_decay = 0.01               # Regularization
```

### Directory Structure
```
gpt2-from-scratch/
├── data_instruct/               # NEW: Instruction dataset shards
│   ├── train/
│   └── val/
├── checkpoints_instruct/        # NEW: Fine-tuning checkpoints
├── runs_instruct/               # NEW: TensorBoard logs
├── src/
│   ├── prepare_instruct_dataset.py    # NEW
│   ├── dataloader_instruct.py         # NEW
│   ├── train_instruct.py              # NEW
│   ├── chat.py                        # NEW
│   └── inference_instruct.py          # NEW
└── start_instruct_training.sh         # NEW
```

---

## ⚠️ Important Notes

1. **Separate Everything**: New folders, new scripts - don't touch existing training code
2. **Load Pre-trained Model**: Start from `checkpoints/best_model.pt`
3. **Lower Learning Rate**: Critical to avoid catastrophic forgetting
4. **Keep It Simple**: No loss masking for now, just basic fine-tuning
5. **Test Early**: Generate samples during training to monitor progress

---

## 🚀 Execution Plan

1. Implement data preparation script
2. Test data formatting on a few samples
3. Implement dataloader
4. Implement training script
5. Start training (8-12 hours estimated)
6. Implement chat interface
7. Test and iterate

---

## ✅ Success Criteria

- [ ] Data downloads and formats correctly
- [ ] Training starts without errors
- [ ] Loss decreases steadily
- [ ] Model generates coherent responses
- [ ] Chat interface works smoothly
- [ ] No interference with existing pre-training code

---

## ✅ IMPLEMENTATION COMPLETE

**Status**: ✅ All code implemented and ready to use
**Implementation Time**: Completed
**Next Step**: Run data preparation and start training

### What's Ready:

1. **Data Preparation**: `python src/prepare_instruct_dataset.py`
2. **Training Script**: `./start_instruct_training.sh` or `./start_instruct_training.sh 4`
3. **Chat Interface**: `./chat.sh`
4. **Documentation**: See `INSTRUCT_TRAINING_GUIDE.md`

### Bug Fixes Applied:
- ✅ Fixed batch size calculation (changed total_batch_size from 128 to 65,536 tokens)
- ✅ Gradient accumulation: 8 steps (65536 / 8192 = 8)
- ✅ All scripts tested for syntax errors

### Ready to Start:
```bash
# Step 1: Prepare data (5-10 minutes)
python src/prepare_instruct_dataset.py

# Step 2: Start training (8-12 hours single GPU)
./start_instruct_training.sh

# Step 3: Chat with your model
./chat.sh
```

**Estimated Training Time**: 8-12 hours on single GPU, 2-4 hours on 4 GPUs

