###
 # @Author: fly
 # @Date: 2024-12-26 16:55:54
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/ex_scripts_A100/llava7b_mistral_origin_A100_v1_2epoch_2A100/eval_chestxray_classify.sh
 # @Description: 
### 

# ========================
# Testing/Evaluation
# ========================
echo "Starting evaluation process..."

python -m llava.run.eval.origin_eval_classify_chestxray \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/llava7b_mistral_sft_v1_2_16 \
    --output-path ./data/chest_xray/Chest-X-ray_llava_origin_val_ans.jsonl \
    --class_path ./data/chest_xray/Chest-X-ray_classes.json \
    --result-file ./result/experiments/Ex_A100/2025_2_16/llava7b_mistral_lora_sft_v1_2epoch_2A100/Chest_Xray_classify_origin.txt \
    --question-file ./data/chest_xray/Chest-X-ray_llava_origin_val.jsonl \
    --inference origin \
    --image-folder /mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali \
    --conv-mode vicuna_v1 


python -m llava.run.eval.origin_eval_classify_chestxray \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/llava7b_mistral_sft_v1_2_16 \
    --output-path ./data/chest_xray/Chest-X-ray_llava_origin_val_ans.jsonl \
    --class_path ./data/chest_xray/Chest-X-ray_classes.json \
    --result-file ./result/experiments/Ex_A100/2025_2_16/llava7b_mistral_lora_sft_v1_2epoch_2A100/Chest_Xray_classify_clip.txt \
    --question-file ./data/chest_xray/Chest-X-ray_llava_origin_val.jsonl \
    --inference clip \
    --image-folder /mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali \
    --conv-mode vicuna_v1 