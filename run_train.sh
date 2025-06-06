accelerate launch --multi_gpu train.py \
  --resolution=256 \
  --output_dir="asv_256_exp1" \
  --train_batch_size=16 \
  --num_epochs=150 \
  --gradient_accumulation_steps=8 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=fp16 \
