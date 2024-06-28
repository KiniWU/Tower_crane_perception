export MODEL_NAME="/home/weights/sdxl_1_0"
export VAE_NAME="/home/weights/madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="/home/tower_crane_data/gen_dataset/333-v3/train/images"

accelerate launch train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="A Modular Integrated Construction lift on the construction site" --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="sdxl-naruto-model" \
  --caption_column="image"











DreamBooth

export MODEL_NAME="/home/weights/sd1-4/CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/home/tower_crane_data/gen_dataset/333-v3/train/images/" # "/home/tower_crane_data/gen_dataset/dreambooth_dataset"
export OUTPUT_DIR="dreambooth_mic"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks mic" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub