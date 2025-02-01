###
 # @Author: fly
 # @Date: 2024-12-26 16:55:54
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/scripts_A100/2025-02-01/llava_mistral_clip_A100_v1_img4_txt6_lr2e5_layer2_3epoch_2A100/eval_chestxray_classify.sh
 # @Description: 
### 

# ========================
# Testing/Evaluation
# ========================
echo "Starting evaluation process..."

python -m llava.run.eval.eval_classify \
    --model-path ./checkpoints/llava_mistral_new_clip_a100_version1_2_1 \
    --result-file ./result/A100/2025_2_1/llava_mistral_clip_A100_v1_img4_txt6_lr2e5_layer2_3epoch_2A100/Chest_Xray_classify.txt \
    --question-file ./data/chest_xray/Chest-X-ray_llava_val.jsonl \
    --image-folder /mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali \
    --conv-mode vicuna_v1 \
    --Imgcls_count 4 \
    --Txtcls_count 6 \
    --hidden_dim 1024 \
    --output_dim 4096 \
    --mlp_type 0 \
    --loss_threshold 0.5 \
    --temperature 0.05 \
    --use_local_loss True \
    --feature_layer 2 \
    --special_tokens_mlp_type 1 \
    --use_ca_loss False \
    --use_cat True

