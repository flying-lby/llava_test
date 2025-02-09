python -m llava.run.train.merge_lora_weights \
    --model-path ./checkpoints/llava-lora-new-clip-v1 \
    --model-base /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --save-model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v1 \
    --Imgcls_count 4 \
    --Txtcls_count 4 \
    --hidden_dim 1024 \
    --output_dim 4096 \
    --img_mlp_type 1 \
    --txt_mlp_type 1 \
    --knowledge_mlp_type 1 \
    --loss_threshold 0.5 \
    --temperature 0.05 \
    --use_local_loss True \
    --feature_layer 2 \
    --special_tokens_mlp_type 1 \
    --use_ca_loss False \
    --use_cat True 
   