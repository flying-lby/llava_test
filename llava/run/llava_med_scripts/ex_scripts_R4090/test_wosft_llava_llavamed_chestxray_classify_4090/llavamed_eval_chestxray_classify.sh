###
 # @Author: fly
 # @Date: 2024-12-26 23:09:38
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/ex_scripts_R4090/test_wosft_llava_llavamed_chestxray_classify_4090/llavamed_eval_chestxray_classify.sh
 # @Description: 
### 
python -m llava.run.eval.origin_eval_classify \
    --model-path /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --output-path ./data/siim/llava7b/siim_llava_origin_val_ans.jsonl \
    --dataset "siim" \
    --result-file ./result/experiments/Ex_R4090/llava7b/siim_classify_origin.txt \
    --inference clip \
    --image-folder "/srv/lby/" \
    --use-cot 1 \
    --conv-mode vicuna_v1 

python -m llava.run.eval.origin_eval_classify \
    --model-path /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --output-path ./data/chest_xray/llava7b/Chest-X-ray_llava_origin_val_ans.jsonl \
    --dataset "chestxray" \
    --result-file ./result/experiments/Ex_R4090/llava7b/Chest_Xray_classify_origin.txt \
    --inference origin \
    --image-folder "/srv/lby/" \
    --conv-mode vicuna_v1 

python -m llava.run.eval.origin_eval_classify \
    --model-path /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --output-path ./data/chexpert/llava7b/chexpert_llava_origin_val_ans.jsonl \
    --dataset "chexpert" \
    --chexpert-subset "False" \
    --result-file ./result/experiments/Ex_R4090/llava7b/chexpert_classify_origin.txt \
    --inference origin \
    --image-folder "/srv/lby/" \
    --conv-mode vicuna_v1 

python -m llava.run.eval.origin_eval_classify \
    --model-path /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --output-path ./data/covid-cxr2/llava7b/covid-cxr2_llava_origin_val_ans.jsonl \
    --dataset "covid-cxr2" \
    --chexpert-subset "False" \
    --result-file ./result/experiments/Ex_R4090/llava7b/covid-cxr2_classify_origin.txt \
    --inference origin \
    --image-folder "/srv/lby/" \
    --conv-mode vicuna_v1 

 python -m llava.run.eval.origin_eval_classify \
    --model-path /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
    --output-path ./data/rsna/llava7b/rsna_llava_origin_val_ans.jsonl \
    --dataset "rsna" \
    --chexpert-subset "False" \
    --result-file ./result/experiments/Ex_R4090/llava7b/rsna_classify_origin.txt \
    --inference origin \
    --image-folder "/srv/lby/" \
    --conv-mode vicuna_v1 


# python -m llava.run.eval.origin_eval_classify_chestxray \
#     --model-path /srv/lby/llava_mistral_7b_official \
#     --output-path ./data/chest_xray/Chest-X-ray_llava_origin_val_ans.jsonl \
#     --dataset "chexpert" \
#     --chexpert-subset "False" \
#     --result-file ./result/experiments/Ex_R4090/llava7b_Chest_Xray_classify_test_origin.txt \
#     --question-file ./data/chest_xray/Chest-X-ray_llava_origin_val.jsonl \
#     --inference origin \
#     --image-folder "/srv/lby/" \
#     --conv-mode vicuna_v1 

# python -m llava.run.eval.origin_eval_classify_chestxray \
#     --model-path /srv/lby/llava_mistral_7b_official \
#     --output-path ./data/chest_xray/Chest-X-ray_llava_origin_val_ans.jsonl \
#     --class_path ./data/chest_xray/Chest-X-ray_classes.json \
#     --result-file ./result/experiments/Ex_R4090/llava7b_Chest_Xray_classify_origin.txt \
#     --question-file ./data/chest_xray/Chest-X-ray_llava_origin_val.jsonl \
#     --inference origin \
#     --image-folder "/srv/lby" \
#     --conv-mode vicuna_v1 

# python -m llava.run.eval.origin_eval_classify_chestxray \
#     --model-path /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
#     --output-path ./data/chest_xray/Chest-X-ray_llava_origin_val_ans.jsonl \
#     --class_path ./data/chest_xray/Chest-X-ray_classes.json \
#     --result-file ./result/experiments/Ex_R4090/llava_med_Chest_Xray_classify_clip.txt \
#     --question-file ./data/chest_xray/Chest-X-ray_llava_origin_val.jsonl \
#     --inference clip \
#     --image-folder "/srv/lby" \
#     --conv-mode vicuna_v1 

# python -m llava.run.eval.origin_eval_classify_chestxray \
#     --model-path /srv/lby/llava_med/llava-med-v1.5-mistral-7b \
#     --output-path ./data/chest_xray/Chest-X-ray_llava_origin_val_ans.jsonl \
#     --class_path ./data/chest_xray/Chest-X-ray_classes.json \
#     --result-file ./result/experiments/Ex_R4090/llava_med_Chest_Xray_classify_origin.txt \
#     --question-file ./data/chest_xray/Chest-X-ray_llava_origin_val.jsonl \
#     --inference origin \
#     --image-folder "/srv/lby" \
#     --conv-mode vicuna_v1 
