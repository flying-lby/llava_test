#!/bin/bash
###
 # @Author: fly
 # @Date: 2024-12-26 16:50:23
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/scripts_A100/2025-02-04/llava_mistral_clip_A100_v1_imlp1_tmlp1_lr2e5_layer2_2epoch_2A100/train_merge_lora.sh
 # @Description: 
### 

# ========================
# Training 
# mlp 3
# ========================
echo "Starting training process..."

deepspeed train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-6 \
    --deepspeed train/zero3.json \
    --model_name_or_path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/llava-med-v1.5-mistral-7b \
    --version v1 \
    --data_path ./data/chest_xray/new_classify_mimic_file_clip.json \
    --image_folder /mnt/nlp-ali/usr/huangwenxuan/home/dataset/srv/lby/physionet.org/files/mimic-cxr-jpg/2.0.0/files \
    --vision_tower /mnt/nlp-ali/usr/huangwenxuan/home/vision_tower/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --mis_mlp_lr 2e-5 \
    --output_dir ./checkpoints/llava-lora-new-clip-A100-version1_2_4 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --Imgcls_count 4 \
    --Txtcls_count 8 \
    --hidden_dim 1024 \
    --output_dim 4096 \
    --img_mlp_type 1 \
    --txt_mlp_type 1 \
    --loss_threshold 0.5 \
    --temperature 0.05 \
    --use_local_loss True \
    --feature_layer 2 \
    --special_tokens_mlp_type 1 \
    --use_ca_loss False \
    --use_cat True
    

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting..."
    exit 1
fi
echo "Training completed successfully."

# ========================
# Merge LoRA
# ========================
echo "Starting merge process..."

python -m llava.run.train.merge_lora_weights \
    --model-path ./checkpoints/llava-lora-new-clip-A100-version1_2_4 \
    --model-base /mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/llava-med-v1.5-mistral-7b \
    --save-model-path ./checkpoints/llava_mistral_new_clip_a100_version1_2_4 \
    --Imgcls_count 4 \
    --Txtcls_count 8 \
    --hidden_dim 1024 \
    --output_dim 4096 \
    --img_mlp_type 1 \
    --txt_mlp_type 1 \
    --loss_threshold 0.5 \
    --temperature 0.05 \
    --use_local_loss True \
    --feature_layer 2 \
    --special_tokens_mlp_type 1 \
    --use_ca_loss False \
    --use_cat True

if [ $? -ne 0 ]; then
    echo "Merge failed. Exiting..."
    exit 1
fi
echo "Merge completed successfully."