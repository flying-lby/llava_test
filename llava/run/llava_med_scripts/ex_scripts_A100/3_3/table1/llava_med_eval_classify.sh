###
 # @Author: fly
 # @Date: 2024-12-26 16:55:54
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/ex_scripts_A100/3_2/table1/llavamed_eval_classify.sh
 # @Description: 
### 

# ========================
# Testing/Evaluation
# ========================
echo "Starting evaluation process..."

# python -m llava.run.eval.origin_eval_classify \
#     --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/llava-med-v1.5-mistral-7b \
#     --output-path ./data/siim/llava_med/siim_llava_origin_val_ans.jsonl \
#     --dataset "siim" \
#     --result-file ./result/experiments/Ex_A100/2025_3_2/llava_med/siim_classify_origin.txt \
#     --inference origin \
#     --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
#     --conv-mode vicuna_v1 


# python -m llava.run.eval.origin_eval_classify \
#     --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/llava-med-v1.5-mistral-7b \
#     --output-path ./data/chexpert/llava_med/chexpert_llava_origin_val_ans.jsonl \
#     --dataset "chexpert" \
#     --chexpert-subset "False" \
#     --result-file ./result/experiments/Ex_A100/2025_3_2/llava_med/chexpert_classify_origin.txt \
#     --inference origin \
#     --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
#     --conv-mode vicuna_v1 


# python -m llava.run.eval.origin_eval_classify \
#     --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/llava-med-v1.5-mistral-7b \
#     --output-path ./data/chexpert/llava_med/chexpertsubset_llava_origin_val_ans.jsonl \
#     --dataset "chexpert" \
#     --chexpert-subset "True" \
#     --result-file ./result/experiments/Ex_A100/2025_3_2/llava_med/chexpertsubset_classify_origin.txt \
#     --inference origin \
#     --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
#     --conv-mode vicuna_v1 

python -m llava.run.eval.origin_eval_classify \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/llava-med-v1.5-mistral-7b \
    --output-path ./data/covid-cxr2/llava_med/covid-cxr2_llava_origin_val_ans.jsonl \
    --dataset "covid-cxr2" \
    --chexpert-subset "True" \
    --result-file ./result/experiments/Ex_A100/2025_3_2/llava_med/covid-cxr2_classify_origin.txt \
    --inference origin \
    --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
    --conv-mode vicuna_v1 

# python -m llava.run.eval.origin_eval_classify \
#     --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/llava-med-v1.5-mistral-7b \
#     --output-path ./data/rsna/llava_med/rsna_llava_origin_val_ans.jsonl \
#     --dataset "rsna" \
#     --chexpert-subset "True" \
#     --result-file ./result/experiments/Ex_A100/2025_3_2/llava_med/rsna_classify_origin.txt \
#     --inference origin \
#     --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
#     --conv-mode vicuna_v1 

# python -m llava.run.eval.origin_eval_classify \
#     --model-path /mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/llava-med-v1.5-mistral-7b \
#     --output-path ./data/chest_xray/llava_med/Chest-X-ray_llava_origin_val_ans.jsonl \
#     --dataset "chestxray" \
#     --result-file ./result/experiments/Ex_A100/2025_3_2/llava_med/Chest_Xray_classify_origin.txt \
#     --inference origin \
#     --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
#     --conv-mode vicuna_v1 