python -m llava.run.train.merge_lora_weights \
    --model-path ./checkpoints/llava-lora-new-clip-v1 \
    --model-base /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --save-model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v1 \
    --ncls_count 6 \
    --hidden_dim 1024 \
    --output_dim 512 \
    --mlp_type 1 \
    --loss_threshold 0.2 \
    --temperature 0.07 \
    --use_local_loss True 

   