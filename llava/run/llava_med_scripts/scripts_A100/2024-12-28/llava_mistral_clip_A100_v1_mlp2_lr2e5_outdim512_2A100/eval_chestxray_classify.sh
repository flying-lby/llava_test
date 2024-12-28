###
 # @Author: fly
 # @Date: 2024-12-26 16:55:54
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/scripts_A100/2024-12-28/llava_mistral_clip_A100_v1_mlp2_lr2e5_outdim512_2A100/eval_chestxray_classify.sh
 # @Description: 
### 

# ========================
# Testing/Evaluation
# ========================
echo "Starting evaluation process..."

python -m llava.run.eval.eval_classify \
    --model-path ./checkpoints/llava_mistral_new_clip_a100_version7_12_28 \
    --result-file ./result/A100/llava_mistral_clip_A100_v1_mlp2_lr2e5_outdim512_2A100/Chest_Xray_classify.txt \
    --question-file ./data/chest_xray/Chest-X-ray_llava_val.jsonl \
    --image-folder /mnt/nlp-ali/usr/zhaizijie/huangwx_ali/zijie_ali \
    --conv-mode vicuna_v1 \
    --ncls_count 4 \
    --hidden_dim 1024 \
    --output_dim 512 \
    --mlp_type 2 \
    --loss_threshold 0.4 \
    --temperature 0.05 \
    --use_local_loss True 


