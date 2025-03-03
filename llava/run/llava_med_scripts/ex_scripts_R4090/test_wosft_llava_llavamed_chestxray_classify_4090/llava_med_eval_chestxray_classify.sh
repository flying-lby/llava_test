###
 # @Author: fly
 # @Date: 2024-12-26 23:09:38
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/ex_scripts_R4090/test_wosft_llava_llavamed_chestxray_classify_4090/llava_med_eval_chestxray_classify.sh
 # @Description: 
### 
python -m llava.run.eval.origin_eval_classify \
    --model-path /srv/lby/llava_mistral_7b_official \
    --output-path ./data/siim/llava7b/siim_llava_origin_test.jsonl \
    --dataset "siim" \
    --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b/siim_classify_origin.txt \
    --inference origin \
    --use-cot 1 \
    --image-folder /srv/lby/ \
    --conv-mode vicuna_v1 
# python -m llava.run.eval.origin_eval_classify_chestxray \
#     --model-path /srv/lby/llava_med/checkpoints/llava_med_mistral_sft_v1  \
#     --output-path ./data/chest_xray/Chest-X-ray_llava_origin_val_ans.jsonl \
#     --class_path ./data/chest_xray/Chest-X-ray_classes.json \
#     --result-file ./result/experiments/Ex_R4090/llava_med_sft_v1_Chest_Xray_classify_clip.txt \
#     --question-file ./data/chest_xray/Chest-X-ray_llava_origin_val.jsonl \
#     --inference clip \
#     --image-folder "/srv/lby" \
#     --conv-mode vicuna_v1 

# python -m llava.run.eval.origin_eval_classify_chestxray \
#     --model-path /srv/lby/llava_med/checkpoints/llava_med_mistral_sft_v1  \
#     --output-path ./data/chest_xray/Chest-X-ray_llava_origin_val_ans.jsonl \
#     --class_path ./data/chest_xray/Chest-X-ray_classes.json \
#     --result-file ./result/experiments/Ex_R4090/llava_med_sft_v1_Chest_Xray_classify_origin.txt \
#     --question-file ./data/chest_xray/Chest-X-ray_llava_origin_val.jsonl \
#     --inference origin \
#     --image-folder "/srv/lby" \
#     --conv-mode vicuna_v1 
