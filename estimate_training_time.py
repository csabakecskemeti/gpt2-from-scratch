#!/usr/bin/env python3
"""
Calculate estimated training time based on current performance
"""

# Current training stats from your output
current_step = 329
step_time_ms_1 = 37213.34  # step 328
step_time_ms_2 = 37854.78  # step 329
avg_step_time_ms = (step_time_ms_1 + step_time_ms_2) / 2

# Training configuration
total_batch_size = 524288
steps_per_epoch = 19073
num_epochs = 5
total_steps = steps_per_epoch * num_epochs

print("=" * 70)
print("TRAINING TIME ESTIMATION")
print("=" * 70)
print()

# Current performance
avg_step_time_sec = avg_step_time_ms / 1000
print(f"📊 Current Performance:")
print(f"   Current step:        {current_step:,}")
print(f"   Avg step time:       {avg_step_time_sec:.2f} seconds ({avg_step_time_ms:.2f} ms)")
print(f"   Tokens/sec:          {total_batch_size / avg_step_time_sec:,.0f}")
print()

# Training configuration
print(f"⚙️  Training Configuration:")
print(f"   Steps per epoch:     {steps_per_epoch:,}")
print(f"   Total epochs:        {num_epochs}")
print(f"   Total steps:         {total_steps:,}")
print(f"   Tokens per step:     {total_batch_size:,}")
print()

# Calculate remaining time
steps_remaining = total_steps - current_step
print(f"⏱️  Time Calculation:")
print(f"   Steps completed:     {current_step:,}")
print(f"   Steps remaining:     {steps_remaining:,}")
print(f"   Progress:            {(current_step / total_steps * 100):.2f}%")
print()

# Total time estimation
time_remaining_sec = steps_remaining * avg_step_time_sec
time_remaining_min = time_remaining_sec / 60
time_remaining_hours = time_remaining_min / 60
time_remaining_days = time_remaining_hours / 24

print(f"⏰ Estimated Time Remaining:")
print(f"   Seconds:             {time_remaining_sec:,.0f}")
print(f"   Minutes:             {time_remaining_min:,.0f}")
print(f"   Hours:               {time_remaining_hours:,.1f}")
print(f"   Days:                {time_remaining_days:.1f}")
print()

# Total training time (from start)
total_time_sec = total_steps * avg_step_time_sec
total_time_hours = total_time_sec / 3600
total_time_days = total_time_hours / 24

print(f"📅 Total Training Time (from step 0):")
print(f"   Hours:               {total_time_hours:,.1f}")
print(f"   Days:                {total_time_days:.1f}")
print()

# Time elapsed so far
time_elapsed_sec = current_step * avg_step_time_sec
time_elapsed_hours = time_elapsed_sec / 3600
time_elapsed_days = time_elapsed_hours / 24

print(f"✅ Time Elapsed So Far:")
print(f"   Hours:               {time_elapsed_hours:.2f}")
print(f"   Days:                {time_elapsed_days:.2f}")
print()

# ETA
print(f"🎯 Summary:")
print(f"   Total training time: ~{total_time_days:.0f} days")
print(f"   Time remaining:      ~{time_remaining_days:.0f} days")
print(f"   Completion at current rate")
print()

# Comparison with earlier estimate
earlier_estimate_days = 34
print(f"📊 Comparison:")
print(f"   Earlier estimate:    {earlier_estimate_days} days")
print(f"   Current estimate:    {total_time_days:.1f} days")
if total_time_days > earlier_estimate_days:
    print(f"   Difference:          +{total_time_days - earlier_estimate_days:.1f} days SLOWER ⚠️")
    print(f"   Reason:              Single GPU (not 4 GPUs)")
else:
    print(f"   Difference:          {total_time_days - earlier_estimate_days:.1f} days FASTER ✅")
print()

# Multi-GPU projection
print(f"💡 Multi-GPU Projection:")
for num_gpus in [2, 4, 8]:
    multi_gpu_days = total_time_days / num_gpus
    print(f"   With {num_gpus} GPUs:          ~{multi_gpu_days:.1f} days")
print()

print("=" * 70)
print("💡 TIP: Run with more GPUs to reduce training time!")
print("   torchrun --standalone --nproc_per_node=4 src/train_improved.py --resume latest --use_tensorboard")
print("=" * 70)

