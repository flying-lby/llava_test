###
 # @Author: fly
 # @Date: 2024-12-26 16:55:54
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/ex_scripts_A100/3_3/table1/llava7b_ft_eval_classify.sh
 # @Description: 
### 

# ========================
# Testing/Evaluation
# ========================
echo "Starting evaluation process..."

python -m llava.run.eval.origin_eval_classify \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/llava7b_mistral_sft_v1_2_16 \
    --output-path ./data/siim/llava7b_ft/siim_llava_origin_val_ans.jsonl \
    --dataset "siim" \
    --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b_ft/siim_classify_origin.txt \
    --inference origin \
    --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
    --conv-mode vicuna_v1 

python -m llava.run.eval.origin_eval_classify \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/llava7b_mistral_sft_v1_2_16 \
    --output-path ./data/covid-cxr2/llava7b_ft/covid-cxr2_llava_origin_val_ans.jsonl \
    --dataset "covid-cxr2" \
    --chexpert-subset "True" \
    --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b_ft/covid-cxr2_classify_origin.txt \
    --inference origin \
    --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
    --conv-mode vicuna_v1 

python -m llava.run.eval.origin_eval_classify \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/llava7b_mistral_sft_v1_2_16 \
    --output-path ./data/rsna/llava7b_ft/rsna_llava_origin_val_ans.jsonl \
    --dataset "rsna" \
    --chexpert-subset "True" \
    --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b_ft/rsna_classify_origin.txt \
    --inference origin \
    --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
    --conv-mode vicuna_v1 


python -m llava.run.eval.origin_eval_classify \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/llava7b_mistral_sft_v1_2_16 \
    --output-path ./data/chexpert/llava7b_ft/chexpert_llava_origin_val_ans.jsonl \
    --dataset "chexpert" \
    --chexpert-subset "False" \
    --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b_ft/chexpert_classify_origin.txt \
    --inference origin \
    --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
    --conv-mode vicuna_v1 


python -m llava.run.eval.origin_eval_classify \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/llava7b_mistral_sft_v1_2_16 \
    --output-path ./data/chexpert/llava7b_ft/chexpertsubset_llava_origin_val_ans.jsonl \
    --dataset "chexpert" \
    --chexpert-subset "True" \
    --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b_ft/chexpertsubset_classify_origin.txt \
    --inference origin \
    --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
    --conv-mode vicuna_v1 

python -m llava.run.eval.origin_eval_classify \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/llava7b_mistral_sft_v1_2_16 \
    --output-path ./data/chest_xray/llava7b_ft/Chest-X-ray_llava_origin_val_ans.jsonl \
    --dataset "chestxray" \
    --result-file ./result/experiments/Ex_A100/2025_3_2/llava7b_ft/Chest_Xray_classify_origin.txt \
    --inference origin \
    --image-folder /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/ \
    --conv-mode vicuna_v1 