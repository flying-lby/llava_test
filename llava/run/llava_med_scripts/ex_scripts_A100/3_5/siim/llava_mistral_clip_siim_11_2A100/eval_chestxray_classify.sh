###
 # @Author: fly
 # @Date: 2024-12-26 16:55:54
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/ex_scripts_A100/3_5/siim/llava_mistral_clip_siim_11_2A100/eval_chestxray_classify.sh
 # @Description: 
### 

# ========================
# Testing/Evaluation
# ========================
echo "Starting evaluation process..."

echo "Starting evaluation process..."

python -m llava.run.eval.eval_classify \
    --model-path /mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/checkpoints/llava_mistral_siim-A100-version11_3_5 \
    --result-folder ./result/experiments/Ex_A100/2025_3_5/siim/siim_11_classify.txt \
    --image-folder "/mnt/nlp-ali/usr/huangwenxuan/home/zijie_ali/libangyan/dataset/" \
    --dataset "siim" \
    --conv-mode vicuna_v1 \
    --Imgcls_count 4 \
    --Txtcls_count 8 \
    --hidden_dim 1024 \
    --output_dim 4096 \
    --img_mlp_type 0 \
    --txt_mlp_type 0 \
    --knowledge_mlp_type 0 \
    --loss_threshold 0.5 \
    --temperature 0.05 \
    --use_local_loss True \
    --feature_layer 2 \
    --special_tokens_mlp_type 1 \
    --use_ca_loss False \
    --use_cat True \
    --Book_choice 1

