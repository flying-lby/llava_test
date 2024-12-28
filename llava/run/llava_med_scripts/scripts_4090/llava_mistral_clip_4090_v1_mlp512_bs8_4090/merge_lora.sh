python -m llava.run.train.merge_lora_weights \
    --model-path /home/lby/llava_med/LLaVA-Med/llava/run/checkpoints/llava-lora-new-clip-version10 \
    --model-base /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --save-model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_ft10 \
    --ncls_count 4 \
    --hidden_dim 1024 \
    --output_dim 4096 \
    --mlp_type 1 \
    --loss_threshold 0.4 \
    --temperature 0.05 \
    --use_local_loss False 

   