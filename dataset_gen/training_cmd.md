python tutorial_train_mic.py \
  --pretrained_model_name_or_path="/home/weights/sd1-5/runwayml/stable-diffusion-v1-5" \
  --image_encoder_path="/home/weights/h94/IP-Adapter/models/image_encoder" \
  --data_json_file="/home/tower_crane_data/gen_dataset/333-v2/mic_only_dataset.json" \
  --data_root_path="" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=0 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --i_drop_rate=0.4\
  --output_dir="/home/weights/trained_models/ip_adapter_mic_only2" \
  --save_steps=200


  --pretrained_ip_adapter_path="/home/weights/h94/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

python tutorial_train_sdxl_mic.py \
  --pretrained_model_name_or_path="/home/weights/sdxl_1_0" \
  --image_encoder_path="/home/weights/h94/IP-Adapter/sdxl_models/image_encoder" \
  --data_json_file="/home/tower_crane_data/gen_dataset/333-v2/mic_only_dataset.json" \
  --data_root_path="" \
  --mixed_precision="fp16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --dataloader_num_workers=0 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --i_drop_rate=0.4\
  --output_dir="/home/weights/trained_models/ip_adapter_sdxl_mic_only" \
  --save_steps=5000
  --num_train_epochs=100

## Multi GPU
accelerate launch --num_processes 2 --multi_gpu --mixed_precision "fp16" \
  tutorial_train_sdxl_mic.py \
  --pretrained_model_name_or_path="/home/weights/sdxl_1_0" \
  --image_encoder_path="/home/weights/h94/IP-Adapter/sdxl_models/image_encoder" \
  --data_json_file="/home/tower_crane_data/gen_dataset/333-v2/mic_only_dataset.json" \
  --data_root_path="" \
  --mixed_precision="fp16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --dataloader_num_workers=0 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --i_drop_rate=0.4\
  --output_dir="/home/weights/trained_models/ip_adapter_sdxl_mic_only" \
  --save_steps=10000 \
  --num_train_epochs=100