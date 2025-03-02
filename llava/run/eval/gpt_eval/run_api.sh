#!/bin/bash
# run_classification.sh
# 说明：
#   此脚本用于运行 run_classification.py 脚本，并通过命令行参数传入配置项。
#   用户可根据需要修改各参数，参数之间使用反斜杠换行以提高可读性。

python3 api_rsna_classification.py \
  --input_file "./datasets/RSNA_Pneumonia/rsna_classify_test.jsonl" \
  --output_file "./datasets/RSNA_Pneumonia/rsna_classify_output.jsonl" \
  --num_threads 128 \
  --model "qwen-plus" \
  --api_key "sk-5ab75e63f3154bba9212ae7667757665" \
  --base_url "https://dashscope.aliyuncs.com/compatible-mode/v1"
