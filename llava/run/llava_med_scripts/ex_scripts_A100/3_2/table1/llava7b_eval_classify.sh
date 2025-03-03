###
 # @Author: fly
 # @Date: 2024-12-26 16:55:54
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/ex_scripts_A100/3_2/table1/llava7b_eval_classify.sh
 # @Description: 
### 

# ========================
# Testing/Evaluation
# ========================
echo "Starting evaluation process..."

python -m llava.run.eval.origin_eval_classify \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava/llava_mistral_7b_official \
    --output-path ./data/siim/llava7b/siim_llava_origin_val_ans.jsonl \
    --dataset "siim" \
    --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b/siim_classify_origin.txt \
    --inference origin \
    --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
    --conv-mode vicuna_v1 


# python -m llava.run.eval.origin_eval_classify \
#     --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava/llava_mistral_7b_official \
#     --output-path ./data/chexpert/llava7b/chexpert_llava_origin_val_ans.jsonl \
#     --dataset "chexpert" \
#     --chexpert-subset "False" \
#     --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b/chexpert_classify_origin.txt \
#     --inference origin \
#     --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
#     --conv-mode vicuna_v1 


# python -m llava.run.eval.origin_eval_classify \
#     --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava/llava_mistral_7b_official \
#     --output-path ./data/chexpert/llava7b/chexpertsubset_llava_origin_val_ans.jsonl \
#     --dataset "chexpert" \
#     --chexpert-subset "True" \
#     --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b/chexpertsubset_classify_origin.txt \
#     --inference origin \
#     --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
#     --conv-mode vicuna_v1 

python -m llava.run.eval.origin_eval_classify \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava/llava_mistral_7b_official \
    --output-path ./data/covid-cxr2/llava7b/covid-cxr2_llava_origin_val_ans.jsonl \
    --dataset "covid-cxr2" \
    --chexpert-subset "True" \
    --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b/covid-cxr2_classify_origin.txt \
    --inference origin \
    --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
    --conv-mode vicuna_v1 

python -m llava.run.eval.origin_eval_classify \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava/llava_mistral_7b_official \
    --output-path ./data/rsna/llava7b/rsna_llava_origin_val_ans.jsonl \
    --dataset "rsna" \
    --chexpert-subset "True" \
    --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b/rsna_classify_origin.txt \
    --inference origin \
    --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
    --conv-mode vicuna_v1 

# python -m llava.run.eval.origin_eval_classify \
#     --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava/llava_mistral_7b_official \
#     --output-path ./data/chest_xray/llava7b/Chest-X-ray_llava_origin_val_ans.jsonl \
#     --dataset "chestxray" \
#     --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b/Chest_Xray_classify_origin.txt \
#     --inference origin \
#     --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
#     --conv-mode vicuna_v1 