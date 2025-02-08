import argparse
import json
import math
import os

import shortuuid
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import torch
from PIL import Image
from tqdm import tqdm
import random

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.clip_llava_builder import load_pretrained_model
from llava.utils import disable_torch_init


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    # Initialize a counter for processed questions
    counter = 0
    max_count = 500  # Set the limit to 5000

    for line in tqdm(questions):
        if counter >= max_count:
            print(f"Reached {max_count} generated answers. Exiting.")
            break

        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        ############## Enable This Part for ImageWikiQA ##############
        # conv.append_message(conv.roles[1], "Let's think step by step.")
        # prompt = conv.get_prompt().replace("</s>", "")
        # print(prompt)

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    # "model_id": model_name,
                    # "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()

        # Increment the counter after each answer is processed
        counter += 1

    ans_file.close()
    
    

def get_metrics():
    # 读取数据
    output_path = './data/eval/test_prompt/Chest-X-ray_llava_val_ans.jsonl'
    answers = [json.loads(line) for line in open(output_path)]

    disease_list = [
        'fibrosis', 'edema', 'pneumothorax', 'cardiomegaly', 'atelectasis', 'nodule',
        'emphysema', 'no finding', 'mass', 'pleural_thickening', 'effusion',
        'infiltration', 'pneumonia', 'hernia', 'consolidation'
    ]

    print(f"Total number of answers: {len(answers)}")
    random.shuffle(answers)
    selected_answers = answers[:1000]

    # 初始化变量
    all_labels = []  # 存储真实标签 (one-hot)
    all_probs = []   # 存储预测概率 (multi-hot)

    for item in selected_answers:
        # 获取真实标签
        labels = ["-".join(item["question_id"].split("-")[1:])]
        labels = [label.lower() for label in labels]
        label_vector = [1 if disease in labels else 0 for disease in disease_list]
        all_labels.append(label_vector)

        # 获取预测文本
        text = item["text"].lower()
        predicted_vector = [1 if disease in text else 0 for disease in disease_list]
        all_probs.append(predicted_vector)

    # 转换为 NumPy 数组
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 逐类别计算准确率
    accuracies = []
    for i in range(len(disease_list)):
        correct = np.sum((all_probs[:, i] == all_labels[:, i]))
        total = len(all_labels)
        accuracies.append(correct / total)

    mean_accuracy = np.mean(accuracies)

    # 逐类别计算 AUC
    auc_scores = []
    for i in range(len(disease_list)):
        try:
            auc_score = roc_auc_score(all_labels[:, i], all_probs[:, i])
            auc_scores.append(auc_score)
        except ValueError:
            auc_scores.append(np.nan)

    mean_auc = np.nanmean(auc_scores)

    # 逐类别计算 AUPRC
    auprc_scores = []
    for i in range(len(disease_list)):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        auprc_score = auc(recall, precision)
        auprc_scores.append(auprc_score)

    mean_auprc = np.mean(auprc_scores)

    # 逐类别计算 F1 分数
    f1_scores = []
    for i in range(len(disease_list)):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        max_f1 = np.max(f1)
        f1_scores.append(max_f1)

    mean_f1 = np.mean(f1_scores)

    # 输出结果
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    for i, accuracy in enumerate(accuracies):
        print(f"Accuracy for class {disease_list[i]}: {accuracy:.4f}")

    print(f"AUC scores per class: {auc_scores}")
    print(f"Mean AUC: {mean_auc:.4f}")

    print(f"AUPRC scores per class: {auprc_scores}")
    print(f"Mean AUPRC: {mean_auprc:.4f}")

    print(f"Mean F1: {mean_f1:.4f}")
    for i, f1 in enumerate(f1_scores):
        print(f"F1 score for class {disease_list[i]}: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/srv/lby/llava_med/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/srv/lby")
    parser.add_argument("--question-file", type=str, default="./data/eval/test_prompt/Chest-X-ray_llava_val.jsonl")
    parser.add_argument("--answers-file", type=str, default="./data/eval/test_prompt/Chest-X-ray_llava_val_ans.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
    get_metrics()