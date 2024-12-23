#!/bin/bash
###
 # @Author: fly
 # @Date: 2024-08-24 14:44:49
 # @FilePath: /llava_med/LLaVA-Med/llava/run/train/finetune_task_lora.sh
 # @Description: 
### 

deepspeed train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 5e-5 \
    --deepspeed train/zero3.json \
    --model_name_or_path /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --version v1 \
    --data_path ./data/train/sft_data/new_classify_mimic_file_clip.json \
    --image_folder /srv/lby/physionet.org/files/mimic-cxr-jpg/2.0.0/files \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --mis_mlp_lr 5e-4 \
    --output_dir ./checkpoints/llava-lora-new-clip-version9 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb
    

# --image_folder /srv/lby/physionet.org/files/mimic-cxr-jpg/2.0.0/files \

# deepspeed llava/train/train_mem.py
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5
#     --deepspeed ./scripts/zero3.json
#     --model_name_or_path ./llava-med-v1.5-mistral-7b
#     --version v1
#     --data_path ./dataSlake/train.json
#     --image_folder ./dataSlake/imgs
#     --vision_tower openai/clip-vit-large-patch14-336
#     --mm_projector_type mlp2x_gelu
#     --mm_vision_select_layer -2
#     --mm_use_im_start_end False
#     --mm_use_im_patch_token False
#     --image_aspect_ratio pad
#     --group_by_modality_length True
#     --bf16 True
#     --output_dir ./checkpoints/llava-version1
#     --num_train_epochs 1
#     --per_device_train_batch_size 4
#     --per_device_eval_batch_size 4
#     --gradient_accumulation_steps 1
#     --evaluation_strategy "no"
#     --save_strategy "steps"
#     --save_steps 50000
#     --save_total_limit 1
#     --learning_rate 2e-4
#     --weight_decay 0.
#     --warmup_ratio 0.03
#     --lr_scheduler_type "cosine"
#     --logging_steps 1
#     --tf32 True
#     --model_max_length 2048
#     --gradient_checkpointing True
#     --dataloader_num_workers 2
#     --lazy_preprocess True
#     --report_to wandb