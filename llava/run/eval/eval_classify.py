import argparse
import json
import math
import os
from torch.nn.utils.rnn import pad_sequence
import shortuuid
import torch
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
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
    # eval_tokenizer_image_token,
)
from llava.model.clip_llava_builder import load_pretrained_model
from llava.utils import disable_torch_init
from dataclasses import dataclass
import argparse
from dataclasses import asdict
from transformers import HfArgumentParser
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, recall_score, f1_score, roc_auc_score

@dataclass
class SparseArguments:
    Imgcls_count: int = 4
    Txtcls_count: int = 4
    hidden_dim: int = 1024
    output_dim: int = 512
    img_mlp_type: int = 1
    txt_mlp_type: int = 1
    knowledge_mlp_type: int = 1
    loss_threshold: float = 0.5
    temperature: float = 0.05
    use_local_loss: bool = False
    feature_layer: int = 1
    special_tokens_mlp_type: int = 1
    use_ca_loss: bool = True
    inference_type: int = 2
    use_cat: bool = True
    use_prompt: bool = True


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args, sparse_args):
    # Model
    # disable_torch_init()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, sparse_args, device_map='cuda:0'
    )
    
    # 加载类别数据
    with open('./data/chest_xray/Chest-X-ray_classes.json', 'r') as f:
        classes = json.load(f)

    # 确保类别数据是一个列表
    categories = [f"This is a chest X-ray showing {category}" for category in classes]

    # 对类别进行编码
    encoded_categories = [tokenizer(category, return_tensors="pt") for category in categories]
    category_ids = pad_sequence([item.input_ids.squeeze(0) for item in encoded_categories], batch_first=True).to(device)
    category_attention_mask = pad_sequence([item.attention_mask.squeeze(0) for item in encoded_categories], batch_first=True).to(device)
    
    # 类别特征向量存储, 只需要计算一次
    global_category_embeddings_cache = []
    local_category_embeddings_cache = []
    for i in range(category_ids.size(0)):
        category_input_ids = category_ids[i].unsqueeze(0)
        category_attention = category_attention_mask[i].unsqueeze(0)
    
        # 获取每个类别的特征向量
        category_output = model.forward(
            input_ids=category_input_ids, 
            attention_mask=category_attention,
            output_hidden_states=True,
            return_emb = True,
            return_dict=True
        )
        # 获取类别特征的最后一个隐藏层并计算均值
        sparse_args_dict = asdict(sparse_args)
        global_category_embedding = category_output.hidden_states[-sparse_args_dict["feature_layer"]][:, -sparse_args_dict["Txtcls_count"]:]
        # local_category_embedding = category_output.hidden_states[-sparse_args_dict["feature_layer"]][:, :-sparse_args_dict["ncls_count"]].mean(dim=1)
        
        global_category_embedding = model.txt_mlp(global_category_embedding)
        global_category_embedding = global_category_embedding.mean(dim=1)
        # local_category_embedding = model.mis_mlp(local_category_embedding)
        
        global_category_embeddings_cache.append(global_category_embedding)
        # local_category_embeddings_cache.append(local_category_embedding)
        
    # 将类别特征向量拼接成 (N, C) 的矩阵，其中N是类别数量，C是特征维度
    global_category_embeddings_cache = torch.cat(global_category_embeddings_cache, dim=0).to(device)
    # local_category_embeddings_cache = torch.cat(local_category_embeddings_cache, dim=0).to(device)
    # # 打印类别特征向量的维度
    print('Global Category embeddings:', global_category_embeddings_cache)   
    # print('Local Category embeddings:', local_category_embeddings_cache)          

    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # questions = random.sample(questions, min(1000, len(questions)))

    # 存储真实标签和预测结果
    all_labels = []  # 真实标签
    all_predictions = []  # 预测标签
    all_probs = []  # 存储类别概率，用于计算 AUC

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        
        # 创建提示语句
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

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda(0)
        )
        attention_mask = (input_ids != tokenizer.pad_token_id).long().cuda(0)
        
        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0].cuda(0)
        
        with torch.inference_mode():
            outputs = model.inference_pipeline(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                global_category_embeddings_cache=global_category_embeddings_cache,
                # local_category_embeddings_cache=local_category_embeddings_cache,
                images=image_tensor.unsqueeze(0).half().cuda(0),
                image_sizes=[image.size],
                use_cache=True,
            )

            # similarity_probs 是一个 (batch_size, num_classes) 的矩阵
            similarity_probs = outputs  # 已经 softmax 过了，得到每个类别的预测概率
            # 获取真实标签
            labels = ["-".join(line["question_id"].split("-")[1:])]  # 获取标签
            labels = [label.lower() for label in labels]  # 转为小写以方便比较

            # 计算该样本的真实标签
            true_labels = torch.zeros(len(classes))  # 假设 `classes` 是类别列表
            for label in labels:
                # 将逗号分隔的多标签拆分为单个类别
                split_labels = [lbl.strip() for lbl in label.split(",")]
                for split_label in split_labels:
                    if split_label in classes:
                        # 在这里设置真实标签为 1，如果该样本属于该类别
                        true_labels[classes.index(split_label)] = 1
                    else:
                        print(f"Warning: {split_label} not found in classes.")

            # 将标签和预测概率存储到全局变量
            all_labels.append(true_labels.cpu().numpy())
            all_probs.append(similarity_probs.cpu().numpy())

    # 将 all_labels 和 all_probs 转换为 numpy 数组
    all_labels = np.array(all_labels)  # shape: (num_samples, num_classes)
    all_probs = np.array(all_probs).squeeze(1)  # shape: (num_samples, num_classes)

    # 初始化准确率列表
    accuracies = []
    result_metrics = {}

    # 计算每个类别的准确率
    for i in range(all_labels.shape[1]):
        # 获取当前类别的精确度、召回率和阈值
        precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        
        # 计算 F1 分数
        f1 = 2 * precision * recall / (precision + recall + 1e-8)  # 避免分母为0
        max_f1_idx = np.argmax(f1)  # 取最大 F1 分数对应的索引
        
        # 选择最大 F1 分数时的阈值
        best_threshold = thresholds[max_f1_idx]
        
        # 根据该阈值对预测值进行二值化（预测为1的样本）
        all_predictions_binary = (all_probs[:, i] >= best_threshold).astype(int)
        
        # 计算该类别的准确率
        accuracy = (all_predictions_binary == all_labels[:, i]).mean()
        accuracies.append(accuracy)

    # 计算 AUC（逐类计算 AUC）
    auc_scores = []
    for i in range(all_labels.shape[1]):  # 对每个类别计算 AUC
        try:
            auc_score = roc_auc_score(all_labels[:, i], all_probs[:, i])
            auc_scores.append(auc_score)
        except ValueError:
            # 如果该类别的标签都为0或1，roc_auc_score会抛出 ValueError
            auc_scores.append(np.nan)

    # 计算 AUPRC（逐类计算 AUPRC）
    auprc_scores = []
    for i in range(all_labels.shape[1]):  # 对每个类别计算 AUPRC
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        auprc_score = auc(recall, precision)  # 计算 AUPRC
        auprc_scores.append(auprc_score)

    # 计算每个类别的精确度、召回率和 F1 分数
    f1_scores = []
    recall_scores = []
    precision_scores = []
    for i in range(all_labels.shape[1]):
        # 计算每个类别的精确度、召回率和 F1 分数
        precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])

        # 计算 F1 分数
        f1 = 2 * precision * recall / (precision + recall + 1e-8)  # 避免分母为0
        max_f1 = np.max(f1)  # 取最大 F1 分数
        f1_scores.append(max_f1)

        # 记录召回率（对应最大 F1 的召回率）
        recall_scores.append(recall[np.argmax(f1)])

        # 记录精确度
        precision_scores.append(precision[np.argmax(f1)])
        
        
    result_metrics["mean_accuracy"] = np.mean(accuracies)
    result_metrics["mean_auc"] = np.nanmean(auc_scores)
    result_metrics["mean_f1"] = np.mean(f1_scores)
    result_metrics["mean_auprc"] = np.mean(auprc_scores)
    result_metrics["mean_recall"] = np.mean(recall_scores)
    result_metrics["mean_precision"] = np.mean(precision_scores)
    
    result_metrics["accuracies_per_class"] = accuracies
    result_metrics["auc_scores_per_class"] = auc_scores
    result_metrics["auprc_scores_per_class"] = auprc_scores
    result_metrics["f1_scores_per_class"] = f1_scores
    result_metrics["recall_scores_per_class"] = recall_scores
    result_metrics["precision_scores_per_class"] = precision_scores

    # 打印所有计算的结果
    print("\n===== Evaluation Metrics =====")
    for key, value in result_metrics.items():
        if isinstance(value, list) or isinstance(value, np.ndarray):  # 打印所有元素
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")
    
    # 检查目录并创建
    result_dir = os.path.dirname(args.result_file)  # 提取文件路径的目录部分
    if result_dir and not os.path.exists(result_dir):  # 如果目录不存在
        os.makedirs(result_dir, exist_ok=True)  # 创建目录

    # 写入文件
    with open(args.result_file, 'w') as f:
        for key, value in result_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Results saved to {args.result_file}")




# def eval_model(args):
#     # Model
#     disable_torch_init()
#     model_path = os.path.expanduser(args.model_path)
#     model_name = get_model_name_from_path(model_path)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map='cuda:0')

#     questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
#     questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
#     answers_file = os.path.expanduser(args.answers_file)
#     os.makedirs(os.path.dirname(answers_file), exist_ok=True)
#     ans_file = open(answers_file, "w")
#     for line in tqdm(questions):
#         idx = line["question_id"]
#         image_file = os.path.join(args.image_folder,line["image"])
        
#         qs = line["text"].replace('<image>', '').strip()
#         cur_prompt = qs
#         if model.config.mm_use_im_start_end:
#             qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
#         else:
#             qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
#         cur_prompt = '<image>' + '\n' + cur_prompt
#         qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
#         cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
        
#         conv = conv_templates[args.conv_mode].copy()
#         conv.append_message(conv.roles[0], qs)
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()

#         input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(0)

#         image = Image.open(os.path.join(args.image_folder, image_file))
#         image_tensor = process_images([image], image_processor, model.config)[0].cuda(0)

#         # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#         # keywords = [stop_str]
#         # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

#         with torch.inference_mode():
#             output_ids = model.generate(
#                 input_ids,
#                 images=image_tensor.unsqueeze(0).half().cuda(0),
#                 do_sample=True if args.temperature > 0 else False,
#                 temperature=args.temperature,
#                 # no_repeat_ngram_size=3,
#                 max_new_tokens=1024,
#                 use_cache=True)

#         outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

#         ans_id = shortuuid.uuid()
#         ans_file.write(json.dumps({"question_id": idx,
#                                    "prompt": cur_prompt,
#                                    "text": outputs,
#                                    "answer_id": ans_id,
#                                    "model_id": model_name,
#                                    "metadata": {}}) + "\n")
#         ans_file.flush()
#     ans_file.close()
    
    
def get_acc():
    # 读取数据
    output_path = './data/eval/test_prompt/Chest-X-ray_llava_val_ans.jsonl'
    answers = [json.loads(line) for line in open(output_path)]

    disease_list = ['fibrosis', 'edema', 'pneumothorax', 'cardiomegaly', 'atelectasis', 'nodule', 'emphysema', 'no finding', 'mass', 'pleural_thickening', 'effusion', 'infiltration', 'pneumonia', 'hernia', 'consolidation']
    print(f"Total number of answers: {len(answers)}")
    # 随机选择 1000 行
    random.shuffle(answers)
    selected_answers = answers[:1000]

    # 初始化变量
    correct_predictions = 0
    total_predictions = len(selected_answers)
    error_count = 0
    error_question_ids = []

    # 遍历每个 answer，提取 labels 和预测类别
    for item in selected_answers:
        # 获取标签（label），labels 可能包含多个标签，以逗号或其他符号分隔
        labels = ["-".join(item["question_id"].split("-")[1:])]  # 获取 label
        labels = [label.lower() for label in labels]  # 转为小写以方便比较

        # 获取预测的 text
        text = item["text"].lower()

        # 尝试在 text 中找到疾病列表中的元素作为预测结果
        predicted_categories = [disease for disease in disease_list if disease in text]

        if predicted_categories:
            predicted_category = predicted_categories[0]  # 假设预测类别为匹配到的第一个疾病
        else:
            # 如果无法提取预测类别，统计为出错
            error_count += 1
            error_question_ids.append(item["question_id"])
            continue  # 跳过此项

        # 检查预测类别是否在 labels 列表中
        if any(predicted_category in label for label in labels):
            correct_predictions += 1
        else:
            # 如果预测错误，统计出错信息
            error_count += 1
            error_question_ids.append(item["question_id"])

    # 计算准确率
    accuracy = (correct_predictions / total_predictions) * 100

    # 输出结果
    print(f"Total labels: {total_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Number of errors: {error_count}")
    print(f"Error question IDs: {error_question_ids}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/srv/lby/llava_med/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--result-file", type=str, default="./result/chest_xray/Chest-X-ray_classify.json")
    parser.add_argument("--question-file", type=str, default="./data/chest_xray/Chest-X-ray_llava_val.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args, remaining_args = parser.parse_known_args()
    
    # Use HfArgumentParser for SparseArguments
    hf_parser = HfArgumentParser(SparseArguments)
    sparse_args, = hf_parser.parse_args_into_dataclasses(remaining_args)

    eval_model(args, sparse_args)
    # get_acc()