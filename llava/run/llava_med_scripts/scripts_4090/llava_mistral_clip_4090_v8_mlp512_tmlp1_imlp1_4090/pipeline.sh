#!/bin/bash
###
 # @Author: fly
 # @Date: 2024-08-24 14:44:49
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/scripts_4090/llava_mistral_clip_4090_v8_mlp512_tmlp1_imlp1_4090/pipeline.sh
 # @Description: 
### 

# ========================
# Training
# ========================
echo "Starting training process..."


#!/bin/bash
###
 # @Author: fly
 # @Date: 2024-12-26 16:50:23
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/ex_scripts_A100/3_5/chest_xray/llava_mistral_clip_chest_xray_100_2A100/train_merge_lora.sh
 # @Description: 
### 

# ========================
# Training 
# mlp 3
# ========================
echo "Starting training process..."

deepspeed train/clip_train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 5e-5 \
    --deepspeed train/zero3.json \
    --model_name_or_path /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --version v1 \
    --data_path ./data/fine_tuning/covid/covid_train_1.json \
    --image_folder /srv/lby/ \
    --vision_tower /srv/lby/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --mis_mlp_lr 5e-5 \
    --output_dir /srv/lby/llava_med/checkpoints/llava-lora-new-clip-v11 \
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
    --report_to none \
    --Imgcls_count 4 \
    --Txtcls_count 8 \
    --hidden_dim 1024 \
    --output_dim 4096 \
    --img_mlp_type 0 \
    --txt_mlp_type 0 \
    --knowledge_mlp_type 0 \
    --loss_threshold 0.5 \
    --temperature 0.05 \
    --use_local_loss True \
    --feature_layer 2 \
    --special_tokens_mlp_type 1 \
    --use_ca_loss False \
    --use_cat True \
    --Book_choice 1
    

if [ $? -ne 0 ]; then
    echo "Training failed. Exiting..."
    exit 1
fi
echo "Training completed successfully."

# ========================
# Merge LoRA
# ========================
echo "Starting merge process..."

python -m llava.run.train.clip_merge_lora_weights \
    --model-path /srv/lby/llava_med/checkpoints/llava-lora-new-clip-v10 \
    --model-base /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --save-model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v10 \
    --Imgcls_count 4 \
    --Txtcls_count 8 \
    --hidden_dim 1024 \
    --output_dim 4096 \
    --img_mlp_type 0 \
    --txt_mlp_type 0 \
    --knowledge_mlp_type 0 \
    --loss_threshold 0.5 \
    --temperature 0.05 \
    --use_local_loss True \
    --feature_layer 2 \
    --special_tokens_mlp_type 1 \
    --use_ca_loss False \
    --use_cat True \
    --Book_choice 0

if [ $? -ne 0 ]; then
    echo "Merge failed. Exiting..."
    exit 1
fi
echo "Merge completed successfully."

python -m llava.run.eval.eval_classify \
    --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v10 \
    --result-folder ./result/R4090/llava-mistral_new_clip_v10/ \
    --image-folder "/srv/lby/" \
    --dataset "chestxray" \
    --conv-mode vicuna_v1 \
    --Imgcls_count 4 \
    --Txtcls_count 8 \
    --hidden_dim 1024 \
    --output_dim 4096 \
    --img_mlp_type 0 \
    --txt_mlp_type 0 \
    --knowledge_mlp_type 0 \
    --loss_threshold 0.5 \
    --temperature 0.05 \
    --use_local_loss True \
    --feature_layer 2 \
    --special_tokens_mlp_type 1 \
    --use_ca_loss False \
    --use_cat True \
    --Book_choice 0