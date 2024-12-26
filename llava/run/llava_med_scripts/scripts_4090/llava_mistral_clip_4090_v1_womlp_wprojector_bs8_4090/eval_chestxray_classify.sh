python -m llava.run.eval.eval_classify \
    --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_ft9 \
    --result-file ./result/R4090/llava-mistral_new_clip_ft9/Chest_Xray_classify.txt \
    --question-file ./data/chest_xray/Chest-X-ray_llava_val.jsonl \
    --image-folder "/srv/lby" \
    --temperature 0 \
    --conv-mode vicuna_v1