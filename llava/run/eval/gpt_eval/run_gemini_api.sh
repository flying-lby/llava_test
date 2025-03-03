#!/bin/bash
# run_classification_gemini.sh
# 说明：
#   此脚本使用 gemini API 对五个数据集进行分类任务，
#   输出文件保存在 ./outputs_api_classify/gemini/ 目录下。

# 定义 gemini 的 API key
gemini='AIzaSyCiUnZYp4syLZvAX4n7tpVMLboiU3C1HsA'

# 创建输出目录
mkdir -p ./outputs_api_classify/gemini

# 对 COVIDx_CXR 数据集进行分类
python3 api_rsna_classification.py \
  --input_file "./data/COVIDx_CXR/COVIDx_CXR_llava_origin_val.jsonl" \
  --output_file "./outputs_api_classify/gemini/COVIDx_CXR_llava_origin_val.jsonl" \
  --output_csv_path "./outputs_api_classify/gemini/covidx_cxr_metric.csv" \
  --num_threads 128 \
  --model_class "gemini" \
  --api_key $gemini

# 对 SIIM_Pneumothorax 数据集进行分类
python3 api_rsna_classification.py \
  --input_file "./data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_origin_val.jsonl" \
  --output_file "./outputs_api_classify/gemini/SIIM_Pneumothorax_llava_origin_val.jsonl" \
  --output_csv_path "./outputs_api_classify/gemini/siim_pneumothorax_metric.csv" \
  --num_threads 128 \
  --model_class "gemini" \
  --api_key $gemini

# 对 Chest-X-ray 数据集进行分类
python3 api_rsna_classification.py \
  --input_file "./data/chest_xray/Chest-X-ray_llava_origin_val.jsonl" \
  --output_file "./outputs_api_classify/gemini/Chest-X-ray_llava_origin_val.jsonl" \
  --output_csv_path "./outputs_api_classify/gemini/chest_xray_metric.csv" \
  --num_threads 128 \
  --model_class "gemini" \
  --api_key $gemini

# 对 chexpert 数据集进行分类
python3 api_rsna_classification.py \
  --input_file "./data/chexpert/chexpert_llava_origin_val.jsonl" \
  --output_file "./outputs_api_classify/gemini/chexpert_llava_origin_val.jsonl" \
  --output_csv_path "./outputs_api_classify/gemini/chexpert_metric.csv" \
  --num_threads 128 \
  --model_class "gemini" \
  --api_key $gemini

# 对 rsna 数据集进行分类
python3 api_rsna_classification.py \
  --input_file "./data/rsna/rsna_pneumonia_llava_origin_val.jsonl" \
  --output_file "./outputs_api_classify/gemini/rsna_pneumonia_llava_origin_val.jsonl" \
  --output_csv_path "./outputs_api_classify/gemini/rsna_metric.csv" \
  --num_threads 128 \
  --model_class "gemini" \
  --api_key $gemini
