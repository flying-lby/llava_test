
###
 # @Author: fly
 # @Date: 2025-02-14 21:18:16
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/ex_scripts_R4090/llava_mistral_clip_4090_v8_mlp512_tmlp1_imlp1_4090/merge_lora.sh
 # @Description: 
### 
python -m llava.run.train.merge_lora_weights \
    --model-path /srv/lby/llava_med/checkpoints/llava-lora-version2 \
    --model-base /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --save-model-path /srv/lby/llava_med/checkpoints/llava-mistral_ft2
   