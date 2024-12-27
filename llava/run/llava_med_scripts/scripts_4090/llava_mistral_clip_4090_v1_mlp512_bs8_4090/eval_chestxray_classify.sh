###
 # @Author: fly
 # @Date: 2024-12-26 23:09:38
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/scripts_4090/llava_mistral_clip_4090_v1_mlp512_bs8_4090/eval_chestxray_classify.sh
 # @Description: 
### 
python -m llava.run.eval.eval_classify \
    --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v1 \
    --result-file ./result/R4090/llava-mistral_new_clip_v1/Chest_Xray_classify.txt \
    --question-file ./data/chest_xray/Chest-X-ray_llava_val.jsonl \
    --image-folder "/srv/lby" \
    --temperature 0 \
    --conv-mode vicuna_v1