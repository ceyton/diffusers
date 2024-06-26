# Set environment variables
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="ossaili/archiflux_1410"

# Run the training script with accelerate
accelerate launch train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  $ENABLE_XFORMERS \
  --resolution=1024 \
  --train_batch_size=1 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
#   --report_to="wandb" \
  --validation_prompt="A modern two-story house with a unique triangular pattern facade, featuring large glass windows on the upper floor. It has a corrugated metal upper exterior and a wooden lower exterior. Surrounded by greenery and residential buildings, the house is situated on a sloped lot under a bright, clear sky." \
  --validation_steps=20 \
  --checkpointing_steps=5000 \
  --output_dir="archiflux-sdxl-v1.0" \
#   --push_to_hub
