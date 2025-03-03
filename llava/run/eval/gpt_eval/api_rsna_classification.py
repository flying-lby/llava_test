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
import openai
import time
import google.generativeai as genai
import pandas as pd

def chat_gemini(image_base64, prompt, api_key):
    """
    调用 Gemini API 进行医学影像分析，使用 genai.generate_content 方法。
    """
    import time  # 确保 time 模块可用
    # genai.configure(api_key=api_key)
    genai.configure(api_key=api_key, transport="rest")  # 强制使用 REST 协议

    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))

    model = genai.get_model('gemini-pro-vision')

    output = None
    for i in range(5):
        try:
            # 调用 generate_content 时传入一个列表，包含 prompt 和图像对象
            response = model.generate_content([prompt, image])
            output = response.text
            break  # 成功获得结果后退出重试循环
        except Exception as e:
            print(e, f"gemini retry {i+1}/3")
            time.sleep(60)
            continue

    return output if output is not None else "Error: No response from Gemini API."


def chat_gpt4o(image_base64, prompt, api_key):
    """
    调用 OpenAI GPT-4o 进行医学影像分析，增加重试机制。
    """
    client = openai.OpenAI(api_key=api_key)
    image_url = f"data:image/jpeg;base64,{image_base64}"

    messages = [
        {"role": "system", "content": "你是一个专业的医学影像分析助手。"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt}
            ],
        },
    ]

    output = None
    for i in range(5):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4096,
                temperature=0.01
            )
            output = completion.choices[0].message.content
            break
        except Exception as e:
            print(e, f"gpt-4o retry {i+1}/3")
            time.sleep(60)
            continue

    return output if output is not None else "Error: No response from GPT-4o API."


def call_model_api(model, image_base64, prompt, api_key):
    """
    根据模型类型选择合适的 API 并进行调用。
    """
    if args.model_class == "qwen":
        # client = OpenAI(api_key=api_key, base_url=args.base_url)
        return chat_qwen(args.model, image_base64, prompt)

    elif args.model_class == "gemini":
        return chat_gemini(image_base64, prompt, api_key)  # Gemini 无需 base_url

    elif args.model_class == "gpt":
        # client = openai.OpenAI(api_key=api_key, base_url=args.base_url)
        return chat_gpt4o(image_base64, prompt, api_key)

    else:
        return "Error: Unsupported model type."



def convert_dicom_to_jpg_bytes(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    # 将像素数据归一化并转换为 uint8
    pixel_array = ds.pixel_array.astype(float)
    pixel_array -= pixel_array.min()
    if pixel_array.max() != 0:
        pixel_array /= pixel_array.max()
    pixel_array = (pixel_array * 255).astype('uint8')
    # 使用 PIL 处理灰度图像并转换为 JPEG
    img = Image.fromarray(pixel_array)
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        jpg_data = output.getvalue()
    return jpg_data

def encode_image(image_path):
    # 如果文件后缀为 .dcm，则先转换为 JPEG
    if image_path.lower().endswith('.dcm'):
        jpg_bytes = convert_dicom_to_jpg_bytes(image_path)
        return base64.b64encode(jpg_bytes).decode("utf-8")
    else:
        # 否则直接读取文件二进制数据
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


def call_model(model, client, messages, max_tokens=4096, temperature=0.01, top_p=0.8, timeout=8000):
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

def chat_qwen(model, image_base64, prompt):
    image_url = f"data:image/jpeg;base64,{image_base64}"
    messages = [
        {
            "role": "system",
            # "content": [{"type": "text", "text": "You are a helpful assistant."}],
            "content": [{"type": "text", "text": "你是一个专业的医学影像分析助手。"}],
        },
        {
            "role": "user",
            "content": [

                # 关键修改：使用image类型直接传递base64
                # {"type": "image", "image": image_base64},
                {
                    "type": "image_url", 
                    "image_url": {
                        # "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                        "url": image_url
                    }
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    output = call_model(
        model, client, messages,
        max_tokens=8092,
        temperature=0.01,
        top_p=0.9,
        timeout=8000
    )
    return output

def generate_classification_prompt(record):
    """
    Generates a prompt for the classification task based on the provided question.
    The model should output only a single letter: either "A" or "B".
    "A" indicates pneumonia, and "B" indicates normal.
    Do not include any additional text or explanation.
    """
    question = record["text"]
    prompt = (
        # f"你是一个专业的图像分析助手，请仔细分析我发给你的图片。\n"
        f"Analyze the following chest X-ray image.\n"
        f"Question: {question}\n\n"
        "Respond with a single letter: 'A' for pneumonia or 'B' for normal. No extra text."
    )
    return prompt


# def generate_classification_prompt(record):
#     """
#     生成分类任务的 prompt，要求模型返回格式为 "label:confidence"
#     """
#     question = record["text"]
#     prompt = (
#         f"Based on the provided chest X-ray image, determine the disease indicated. The question is:\n"
#         f"{question}\n\n"
#         "Output only the answer in the exact format 'label:confidence'. "
#         "For example, if you think the image shows a normal chest X-ray with 85% confidence, you should output:\n"
#         "normal:0.85\n"
#         "Do not include any additional text or explanations."
#     )
#     return prompt

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
    Process a single record:
      1. Read and encode the image (DICOM converted to PNG via encode_image);
      2. Generate the classification prompt;
      3. Call the model and parse the answer, which is expected to be only "A" or "B";
      4. Map the answer to the predicted label and set a default confidence.
         Here, "A" corresponds to pneumonia (confidence 0.0) and "B" to normal (confidence 1.0).
      5. Extract the true label from the record.
    """
    image_path = record["image"]
    # Using the given image path directly; ensure the path is correct.
    image_full_path = f"{image_path}"
    encoded_image = encode_image(image_full_path)
    prompt = generate_classification_prompt(record)
    # response = chat("qwen-vl-max-latest", encoded_image, prompt)
    # response = chat_qwen(args.model, encoded_image, prompt)
    response = call_model_api(args.model, encoded_image, prompt, args.api_key)
    
    try:
        answer = response.strip().upper()  # Expecting "A" or "B"
        if answer == "A":
            predicted_label = "pneumonia"
            predicted_confidence = 0.0  # 0 indicates pneumonia
        elif answer == "B":
            predicted_label = "normal"
            predicted_confidence = 1.0  # 1 indicates normal
        else:
            predicted_label = "error"
            predicted_confidence = 0.5
    except Exception as e:
        predicted_label = "error"
        predicted_confidence = 0.5
    # Extract true label from record (assuming the correct label has value 1)
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

    global client

    if args.model_class == "qwen":
        args.model = "qwen-vl-max-latest"
        args.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    elif args.model_class == "gemini":
        args.model = "gemini-pro-vision"

    elif args.model_class == "gpt":
        args.model = "gpt-4o"
        args.base_url = "https://api.openai.com/v1"
        client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)

    else:
        return "Error: Unsupported model type."
    
    print(args)

    # 测试调用：如果使用的是 Gemini，则调用 chat_gemini 进行测试
    if args.model_class == "gemini":
        test_prompt = "图中描绘的是什么景象?"
        # 假设这里有一个测试图片 URL（或加载本地图片并编码）
        test_image_path = "./image.png"
        encoded_image = encode_image(test_image_path)
        test_response = chat_gemini(encoded_image, test_prompt, args.api_key)
        print("Gemini test response:", test_response)
    else:
        # 对于 Qwen 和 GPT-4o，调用 OpenAI client 的接口
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                            },
                        },
                        {"type": "text", "text": "图中描绘的是什么景象?"},
                    ],
                },
            ],
        )
        print(completion.choices[0].message.content)

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


    # 计算评价指标
    y_true = []
    y_pred = []
    y_score = []
    for res in results:
        # Define positive class as "normal": 1; pneumonia as 0.
        true = 1 if res["true_label"] == "normal" else 0
        pred = 1 if res["predicted_label"] == "normal" else 0
        score = res["predicted_confidence"]  # Now is 1.0 (normal) or 0.0 (pneumonia)
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
    # print("Confusion Matrix:\n", cm)

    # 保存指标到 CSV
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "F1-Score"],
        "Value": [acc, auc, f1]
    })
    metrics_df.to_csv(args.output_csv_path, index=False)
    print(f"Metrics saved to {args.output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RSNA Pneumonia Classification.")
    parser.add_argument("--input_file", type=str, default="./datasets/RSNA_Pneumonia/rsna_classify_test.jsonl", 
                        help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, default="./datasets/RSNA_Pneumonia/rsna_classify_output.jsonl", 
                        help="Path to the output JSONL file.")
    parser.add_argument("--num_threads", type=int, default=128, 
                        help="Number of threads to use.")
    parser.add_argument("--model_class", type=str, default="qwen", 
                        help="Model class.")
    parser.add_argument("--model", type=str, default="qwen-vl-max-latest", 
                        help="Model name.")
    parser.add_argument("--api_key", type=str, default="sk-5ab75e63f3154bba9212ae7667757665", 
                        help="API Key.")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                        help="Base URL for the API service.")
    parser.add_argument("--output_csv_path", type=str, default="./datasets/RSNA_Pneumonia/metrics.csv",
                    help="Path to the output CSV file for metrics.")


    args = parser.parse_args()

    main()
