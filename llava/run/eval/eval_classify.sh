python -m llava.run.eval.eval_classify \
    --model-path /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_ft9 \
    --question-file ./data/eval/test_prompt/Chest-X-ray_llava_val.jsonl \
    --image-folder "/srv/lby" \
    --answers-file ./data/eval/test_prompt/Chest-X-ray_llava_val_ans.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1