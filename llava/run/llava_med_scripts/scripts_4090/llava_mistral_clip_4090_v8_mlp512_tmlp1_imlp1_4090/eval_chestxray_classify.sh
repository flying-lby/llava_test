###
 # @Author: fly
 # @Date: 2024-12-26 23:09:38
 # @FilePath: /llava_med/LLaVA-Med/llava/run/llava_med_scripts/scripts_4090/llava_mistral_clip_4090_v8_mlp512_tmlp1_imlp1_4090/eval_chestxray_classify.sh
 # @Description: 
### 
# python -m llava.run.eval.eval_classify \
#     --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v9 \
#     --result-folder ./result/R4090/llava-mistral_new_clip_v9/ \
#     --image-folder "/srv/lby/" \
#     --dataset "siim" \
#     --conv-mode vicuna_v1 \
#     --Imgcls_count 4 \
#     --Txtcls_count 8 \
#     --hidden_dim 1024 \
#     --output_dim 4096 \
#     --img_mlp_type 0 \
#     --txt_mlp_type 0 \
#     --knowledge_mlp_type 0 \
#     --loss_threshold 0.5 \
#     --temperature 0.07 \
#     --use_local_loss True \
#     --feature_layer 2 \
#     --special_tokens_mlp_type 1 \
#     --use_ca_loss False \
#     --use_cat True \
#     --Book_choice 0

# python -m llava.run.eval.eval_classify \
#     --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v9 \
#     --result-folder ./result/R4090/llava-mistral_new_clip_v9/ \
#     --image-folder "/srv/lby/" \
#     --dataset "chestxray" \
#     --conv-mode vicuna_v1 \
#     --Imgcls_count 4 \
#     --Txtcls_count 8 \
#     --hidden_dim 1024 \
#     --output_dim 4096 \
#     --img_mlp_type 0 \
#     --txt_mlp_type 0 \
#     --knowledge_mlp_type 0 \
#     --loss_threshold 0.5 \
#     --temperature 0.05 \
#     --use_local_loss True \
#     --feature_layer 2 \
#     --special_tokens_mlp_type 1 \
#     --use_ca_loss False \
#     --use_cat True

# python -m llava.run.eval.eval_classify \
#     --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v9 \
#     --result-folder ./result/R4090/llava-mistral_new_clip_v9/ \
#     --image-folder "/srv/lby/" \
#     --chexpert-subset "False" \
#     --dataset "chexpert" \
#     --conv-mode vicuna_v1 \
#     --Imgcls_count 4 \
#     --Txtcls_count 8 \
#     --hidden_dim 1024 \
#     --output_dim 4096 \
#     --img_mlp_type 0 \
#     --txt_mlp_type 0 \
#     --knowledge_mlp_type 0 \
#     --loss_threshold 0.5 \
#     --temperature 0.01 \
#     --use_local_loss True \
#     --feature_layer 2 \
#     --special_tokens_mlp_type 1 \
#     --use_ca_loss False \
#     --use_cat True \
#     --Book_choice 1

python -m llava.run.eval.eval_classify \
    --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v9 \
    --result-folder ./result/R4090/llava-mistral_new_clip_v9/ \
    --image-folder "/srv/lby/" \
    --dataset "rsna" \
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
    --Book_choice 0

# python -m llava.run.eval.eval_classify \
#     --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v9 \
#     --result-folder ./result/R4090/llava-mistral_new_clip_v9/ \
#     --image-folder "/srv/lby/" \
#     --subdata "unseen" \
#     --dataset "padchest" \
#     --conv-mode vicuna_v1 \
#     --Imgcls_count 4 \
#     --Txtcls_count 8 \
#     --hidden_dim 1024 \
#     --output_dim 4096 \
#     --img_mlp_type 0 \
#     --txt_mlp_type 0 \
#     --knowledge_mlp_type 0 \
#     --loss_threshold 0.5 \
#     --temperature 0.05 \
#     --use_local_loss True \
#     --feature_layer 2 \
#     --special_tokens_mlp_type 1 \
#     --use_ca_loss False \
#     --use_cat True

# python -m llava.run.eval.eval_classify \
#     --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v9 \
#     --result-folder ./result/R4090/llava-mistral_new_clip_v9/ \
#     --image-folder "/srv/lby/" \
#     --subdata "seen" \
#     --dataset "padchest" \
#     --conv-mode vicuna_v1 \
#     --Imgcls_count 4 \
#     --Txtcls_count 8 \
#     --hidden_dim 1024 \
#     --output_dim 4096 \
#     --img_mlp_type 0 \
#     --txt_mlp_type 0 \
#     --knowledge_mlp_type 0 \
#     --loss_threshold 0.5 \
#     --temperature 0.05 \
#     --use_local_loss True \
#     --feature_layer 2 \
#     --special_tokens_mlp_type 1 \
#     --use_ca_loss False \
#     --use_cat True

# python -m llava.run.eval.eval_classify \
#     --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v9 \
#     --result-folder ./result/R4090/llava-mistral_new_clip_v9/ \
#     --image-folder "/srv/lby/" \
#     --subdata "rare" \
#     --dataset "padchest" \
#     --conv-mode vicuna_v1 \
#     --Imgcls_count 4 \
#     --Txtcls_count 8 \
#     --hidden_dim 1024 \
#     --output_dim 4096 \
#     --img_mlp_type 0 \
#     --txt_mlp_type 0 \
#     --knowledge_mlp_type 0 \
#     --loss_threshold 0.5 \
#     --temperature 0.05 \
#     --use_local_loss True \
#     --feature_layer 2 \
#     --special_tokens_mlp_type 1 \
#     --use_ca_loss False \
#     --use_cat True \
#     --Book_choice 0


# python -m llava.run.eval.eval_classify \
#     --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v9 \
#     --result-file ./result/R4090/llava-mistral_new_clip_v9/rsna_classify.txt \
#     --question-file ./data/rsna/rsna_pneumonia_llava.jsonl \
#     --image-folder "/srv/lby" \
#     --conv-mode vicuna_v1 \
#     --Imgcls_count 4 \
#     --Txtcls_count 8 \
#     --hidden_dim 1024 \
#     --output_dim 4096 \
#     --img_mlp_type 0 \
#     --txt_mlp_type 0 \
#     --knowledge_mlp_type 0 \
#     --loss_threshold 0.5 \
#     --temperature 0.05 \
#     --use_local_loss True \
#     --feature_layer 2 \
#     --special_tokens_mlp_type 1 \
#     --use_ca_loss False \
#     --use_cat True

# python -m llava.run.eval.eval_classify \
#     --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v9 \
#     --result-file ./result/R4090/llava-mistral_new_clip_v9/chexpert_classify.txt \
#     --question-file ./data/chexpert/chexpert_llava_val.jsonl \
#     --image-folder "/srv/lby" \
#     --conv-mode vicuna_v1 \
#     --Imgcls_count 4 \
#     --Txtcls_count 8 \
#     --hidden_dim 1024 \
#     --output_dim 4096 \
#     --img_mlp_type 0 \
#     --txt_mlp_type 0 \
#     --knowledge_mlp_type 0 \
#     --loss_threshold 0.5 \
#     --temperature 0.05 \
#     --use_local_loss True \
#     --feature_layer 2 \
#     --special_tokens_mlp_type 1 \
#     --use_ca_loss False \
#     --use_cat True

# python -m llava.run.eval.origin_eval_classify_chestxray \
#     --model-path /srv/lby/llava_med/checkpoints/llava_med_mistral_sft_v1  \
#     --output-path ./data/chest_xray/Chest-X-ray_llava_origin_val_ans.jsonl \
#     --class_path ./data/chest_xray/Chest-X-ray_classes.json \
#     --result-file ./result/experiments/Ex_R4090/llava_med_sft_v1_Chest_Xray_classify_origin.txt \
#     --question-file ./data/chest_xray/Chest-X-ray_llava_origin_val.jsonl \
#     --inference origin \
#     --image-folder "/srv/lby" \
#     --conv-mode vicuna_v1 

# python -m llava.run.eval.eval_classify_chestxray \
#     --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_v9 \
#     --result-file ./result/R4090/llava-mistral_new_clip_v9/Chest_Xray_classify.txt \
#     --question-file ./data/chest_xray/Chest-X-ray_llava_val.jsonl \
#     --image-folder "/srv/lby" \
#     --conv-mode vicuna_v1 \
#     --Imgcls_count 4 \
#     --Txtcls_count 8 \
#     --hidden_dim 1024 \
#     --output_dim 4096 \
#     --img_mlp_type 0 \
#     --txt_mlp_type 0 \
#     --knowledge_mlp_type 0 \
#     --loss_threshold 0.5 \
#     --temperature 0.05 \
#     --use_local_loss True \
#     --feature_layer 2 \
#     --special_tokens_mlp_type 1 \
#     --use_ca_loss False \
#     --use_cat True
    