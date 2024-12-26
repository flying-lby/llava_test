python -m llava.run.train.merge_lora_weights \
    --model-path /home/lby/llava_med/LLaVA-Med/llava/run/checkpoints/llava-lora-new-clip-version10 \
    --model-base /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --save-model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_ft10

   