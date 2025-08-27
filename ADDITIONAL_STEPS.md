# Additional Steps

## More dependencies

```
pip install pillow opencv-python tqdm matplotlib seaborn pandas scikit-learn sentencepiece protobuf
pip install "trl>=0.14,<0.15" \
            "transformers>=4.44" \
            "accelerate>=0.33" \
            "peft>=0.11"
pip install liger-kernel
pip install "torchvision==0.19.0+cu121" --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes
```

## Run command

```
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node 1 src/grpo_grounding.py \
  --output_dir runs/gtatest1 \
  --model_name_or_path "HelloKKMe/GTA1-7B" \
  --dataset_name "data/captures.jsonl" \
  --image_root "./data" \
  --max_prompt_length 512 \
  --max_completion_length 32 \
  --num_generations 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --freeze_vision_modules true \
  --reward_funcs accuracy \
  --beta 0 \
  --max_pixels $((1280*1024)) \
  --dataloader_num_workers 0 \
  --logging_steps 1 \
  --max_steps 100 \
  --bf16 \
  --attn_implementation eager \
  --save_steps 50 \
  --save_total_limit 1 \
  --save_only_model false
```