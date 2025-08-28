#!/bin/bash

# Training script for GTA1 project
# This script runs the GRPO training with the specified parameters
# Supports both Linux/WSL2 (CUDA) and macOS (CPU/MPS)

set -e  # Exit on any error

# Ensure virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "🚨 Virtual environment not activated. Activating .venv..."
    source .venv/bin/activate
fi

echo "🚀 Starting GTA1 training..."

# Detect platform and set device
OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Darwin" ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        echo "🍎 Apple Silicon Mac detected - using MPS acceleration"
        export PYTORCH_ENABLE_MPS_FALLBACK=1
        DEVICE_ARG=""  # Let PyTorch auto-detect MPS
        PRECISION_ARG="--fp16"  # Use fp16 instead of bf16 on macOS
    else
        echo "🍎 Intel Mac detected - using CPU"
        DEVICE_ARG=""  # CPU only
        PRECISION_ARG=""  # No mixed precision on CPU
    fi
    # Reduce batch size and other parameters for macOS
    BATCH_SIZE=2
    GRAD_ACCUM=2
    NUM_GENERATIONS=2
    MAX_PIXELS=$((640*512))  # Reduced resolution
    echo "⚙️  Using reduced settings for macOS:"
    echo "   - Batch size: $BATCH_SIZE"
    echo "   - Gradient accumulation: $GRAD_ACCUM" 
    echo "   - Num generations: $NUM_GENERATIONS"
    echo "   - Max pixels: $MAX_PIXELS"
    echo "   - Precision: ${PRECISION_ARG:-fp32}"
else
    # Linux/WSL2 - use original CUDA settings
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    echo "📱 Using CUDA device: $CUDA_VISIBLE_DEVICES"
    BATCH_SIZE=2
    GRAD_ACCUM=8
    NUM_GENERATIONS=2
    MAX_PIXELS=$((1280*1024))
    PRECISION_ARG="--bf16"  # Use bf16 on CUDA
fi

# Run training with platform-appropriate settings
torchrun --nproc_per_node 1 src/grpo_grounding.py \
  --output_dir runs/gtatest1 \
  --model_name_or_path "HelloKKMe/GTA1-7B" \
  --dataset_name "data/captures.jsonl" \
  --image_root "./data" \
  --max_prompt_length 512 \
  --max_completion_length 32 \
  --num_generations $NUM_GENERATIONS \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --freeze_vision_modules true \
  --reward_funcs accuracy \
  --beta 0 \
  --max_pixels $MAX_PIXELS \
  --dataloader_num_workers 0 \
  --logging_steps 1 \
  --max_steps 100 \
  $PRECISION_ARG \
  --attn_implementation eager \
  --save_steps 50 \
  --save_total_limit 1 \
  --save_only_model false

echo "✅ Training completed!"
