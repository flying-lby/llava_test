import json
import re
import concurrent.futures
import threading
import argparse
from openai import OpenAI
from tqdm import tqdm
import base64
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import pydicom
import io
from PIL import Image

def convert_dicom_to_png_bytes(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    # 将像素数据归一化并转换为uint8
    pixel_array = ds.pixel_array.astype(np.float32)
    pixel_array -= pixel_array.min()
    pixel_array /= pixel_array.max()
    pixel_array = (pixel_array * 255).astype(np.uint8)
    # PIL处理灰度图像
    img = Image.fromarray(pixel_array)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        png_data = output.getvalue()
    return png_data

def encode_image(image_path):
    # 这里以读取二进制图片文件为例
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_model(model, client, messages, max_tokens=4096, temperature=0.01, top_p=0.1, timeout=2000):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
    )
    return completion.choices[0].message.content

def chat(model, image, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}"
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    output = call_model(model, client_qwen, messages, max_tokens=8092, temperature=1, top_p=0.001, timeout=2000)
    return output

def generate_classification_prompt(record):
    """
    生成分类任务的 prompt，要求模型返回格式为 "label:confidence"
    """
    question = record["text"]
    prompt = (
        f"Based on the provided chest X-ray image, determine the disease indicated. The question is:\n"
        f"{question}\n\n"
        "Output only the answer in the exact format 'label:confidence'. "
        "For example, if you think the image shows a normal chest X-ray with 85% confidence, you should output:\n"
        "normal:0.85\n"
        "Do not include any additional text or explanations."
    )
    return prompt

def collect_tasks(input_file):
    """
    读取输入 JSONL 文件，每行包含 image, text, label 字段，返回任务列表。
    """
    tasks = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tasks.append(data)
    return tasks

def process_task(record):
    """
    处理单个记录：
      1. 根据图片路径读取图片并进行 base64 编码；
      2. 生成分类任务的 prompt；
      3. 调用 qwen 模型获得预测结果，并解析出标签和置信度；
      4. 从 record 中提取真实标签（假设 label 字段为字典，正确标签对应值为1）。
    """
    image_path = record["image"]
    # 构造完整图片路径，根据实际情况修改
    image_full_path = f"{image_path}"
    encoded_image = encode_image(image_full_path)
    prompt = generate_classification_prompt(record)
    response = chat(model_qwen, encoded_image, prompt)
    try:
        parts = response.strip().split(':')
        predicted_label = parts[0].strip().lower()
        predicted_confidence = float(parts[1].strip())
    except Exception as e:
        predicted_label = "error"
        predicted_confidence = 0.0
    # 获取真实标签，假设 label 为字典且正确标签对应值为1
    true_label = None
    for k, v in record["label"].items():
        if v == 1:
            true_label = k.lower()
            break
    return {
        "true_label": true_label,
        "predicted_label": predicted_label,
        "predicted_confidence": predicted_confidence,
        "response": response,
        "image": record["image"],
        "text": record["text"]
    }

def main():
    parser = argparse.ArgumentParser(description="Run RSNA Pneumonia Classification.")
    parser.add_argument("--input_file", type=str, default="./datasets/RSNA_Pneumonia/rsna_classify_test.jsonl", 
                        help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, default="./datasets/RSNA_Pneumonia/rsna_classify_output.jsonl", 
                        help="Path to the output JSONL file.")
    parser.add_argument("--num_threads", type=int, default=128, 
                        help="Number of threads to use.")
    parser.add_argument("--model", type=str, default="qwen-plus", 
                        help="Model name.")
    parser.add_argument("--api_key", type=str, default="sk-5ab75e63f3154bba9212ae7667757665", 
                        help="API Key.")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                        help="Base URL for the API service.")

    args = parser.parse_args()

    global client, client_qwen, model_qwen
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    client_qwen = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    model_qwen = args.model

    input_file = args.input_file
    output_file = args.output_file
    num_threads = args.num_threads

    tasks = collect_tasks(input_file)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_task, record) for record in tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print("Error in task:", e)

    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # 计算评价指标：Accuracy、AUC 和 Confusion Matrix
    y_true = []
    y_pred = []
    y_score = []
    for res in results:
        # 假设正类定义为 "normal"，负类为 "pneumonia"
        true = 1 if res["true_label"] == "normal" else 0
        pred = 1 if res["predicted_label"] == "normal" else 0
        # score 定义：若预测为 normal，score 为置信度；若预测为 pneumonia，则 score 为 1-置信度
        score = res["predicted_confidence"] if res["predicted_label"] == "normal" else 1 - res["predicted_confidence"]
        y_true.append(true)
        y_pred.append(pred)
        y_score.append(score)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)
    print("Accuracy:", acc)
    print("AUC:", auc)
    print("F1:", f1)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
