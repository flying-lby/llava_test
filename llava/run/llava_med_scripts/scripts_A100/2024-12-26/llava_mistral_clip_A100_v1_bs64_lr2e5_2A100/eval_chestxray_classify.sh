###
 # @Author: fly
 # @Date: 2024-12-26 16:55:54
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/scripts_A100/llava_mistral_clip_A100_v1_bs64_lr2e5_2A100/eval_chestxray_classify.sh
 # @Description: 
### 

# ========================
# Testing/Evaluation
# ========================
echo "Starting evaluation process..."

python -m llava.run.eval.eval_classify \
    --model-path ./checkpoints/llava_mistral_new_clip_a100_version2 \
    --result-file ./result/A100/llava_mistral_clip_A100_v1_bs64_lr2e5_2A100/Chest_Xray_classify.txt \
    --question-file ./data/chest_xray/Chest-X-ray_llava_val.jsonl \
    --image-folder /mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali \
    --temperature 0 \
    --conv-mode vicuna_v1


