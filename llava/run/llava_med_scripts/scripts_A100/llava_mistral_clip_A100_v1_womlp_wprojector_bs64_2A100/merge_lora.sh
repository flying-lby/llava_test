# ========================
# Merge LoRA
# ========================
echo "Starting merge process..."

python -m llava.run.train.merge_lora_weights \
    --model-path ./checkpoints/llava-lora-new-clip-A100-version1 \
    --model-base /mnt/nlp-ali/usr/huangwenxuan/home/official_llava_med/llava-med-v1.5-mistral-7b \
    --save-model-path ./checkpoints/llava_mistral_new_clip_a100_version1

if [ $? -ne 0 ]; then
    echo "Merge failed. Exiting..."
    exit 1
fi
echo "Merge completed successfully."