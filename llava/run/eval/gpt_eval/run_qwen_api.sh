#!/bin/bash
# run_classification_qwen.sh
# 说明：
#   此脚本使用 qwen API 对五个数据集进行分类任务，
#   输出文件保存在 ./outputs_api_classify/qwen/ 目录下。

# 定义 qwen 的 API key
qwen='sk-5ab75e63f3154bba9212ae7667757665'

# 创建输出目录
mkdir -p ./outputs_api_classify/qwen

# 对 COVIDx_CXR 数据集进行分类
python3 api_rsna_classification.py \
  --input_file "./data/COVIDx_CXR/COVIDx_CXR_llava_origin_val.jsonl" \
  --output_file "./outputs_api_classify/qwen/COVIDx_CXR_llava_origin_val.jsonl" \
  --output_csv_path "./outputs_api_classify/qwen/covid_metric.csv" \
  --num_threads 128 \
  --model_class "qwen" \
  --api_key $qwen

# 对 SIIM_Pneumothorax 数据集进行分类
python3 api_rsna_classification.py \
  --input_file "./data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_origin_val.jsonl" \
  --output_file "./outputs_api_classify/qwen/SIIM_Pneumothorax_llava_origin_val.jsonl" \
  --output_csv_path "./outputs_api_classify/qwen/siim_metric.csv" \
  --num_threads 128 \
  --model_class "qwen" \
  --api_key $qwen

# 对 Chest-X-ray 数据集进行分类
python3 api_rsna_classification.py \
  --input_file "./data/chest_xray/Chest-X-ray_llava_origin_val.jsonl" \
  --output_file "./outputs_api_classify/qwen/Chest-X-ray_llava_origin_val.jsonl" \
  --output_csv_path "./outputs_api_classify/qwen/chest_metric.csv" \
  --num_threads 128 \
  --model_class "qwen" \
  --api_key $qwen

# 对 chexpert 数据集进行分类
python3 api_rsna_classification.py \
  --input_file "./data/chexpert/chexpert_llava_origin_val.jsonl" \
  --output_file "./outputs_api_classify/qwen/chexpert_llava_origin_val.jsonl" \
  --output_csv_path "./outputs_api_classify/qwen/chexpert_metric.csv" \
  --num_threads 128 \
  --model_class "qwen" \
  --api_key $qwen

# 对 rsna 数据集进行分类
python3 api_rsna_classification.py \
  --input_file "./data/rsna/rsna_pneumonia_llava_origin_val.jsonl" \
  --output_file "./outputs_api_classify/qwen/rsna_pneumonia_llava_origin_val.jsonl" \
  --output_csv_path "./outputs_api_classify/qwen/rsna_metric.csv" \
  --num_threads 128 \
  --model_class "qwen" \
  --api_key $qwen