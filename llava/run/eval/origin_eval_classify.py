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
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from dataclasses import dataclass
import argparse
from dataclasses import asdict
from transformers import HfArgumentParser
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, recall_score, f1_score, roc_auc_score
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from typing import List
import pydicom
from skimage import exposure

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_classes(args):
      
    # 加载类别数据
    chexray14_cls = ["fibrosis","edema","pneumothorax","cardiomegaly","atelectasis","nodule","emphysema","no finding",
                     "mass","pleural_thickening","effusion","infiltration","pneumonia","hernia","consolidation"]  #Fibrosis seldom appears in MIMIC_CXR and is divided into the 'tail_abnorm_obs' entitiy.  
    
    if args.dataset == 'chexpert':
        chexpert_subset = args.chexpert_subset

        if chexpert_subset == 'False':
            chexpert_cls = [
            'no finding', 'enlarged cardiomediastinum', 'cardiomegaly', 
            'lung opacity', 'lung lesion', 'edema', 'consolidation', 
            'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion', 
            'pleural other', 'fracture', 'support devices']
        else:
            chexpert_cls = ['cardiomegaly','edema', 'consolidation', 'atelectasis','pleural effusion']

    siim_cls = ['pneumothorax', 'non-pneumothorax']
    rsna_cls = ['pneumonia','normal']
    covid_cls = ['covid19','non-covid19']

    
    original_class = [
                'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
                'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
                'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
                'tail_abnorm_obs', 'excluded_obs'
            ]
    
    if args.dataset == 'chestxray':
        dataset_cls = chexray14_cls
        question_file = './data/chest_xray/Chest-X-ray_llava_origin_val.jsonl'
    elif args.dataset == 'chexpert':
        dataset_cls = chexpert_cls
        question_file = './data/chexpert/chexpert_llava_origin_val.jsonl'
    elif args.dataset == 'siim':
        dataset_cls = siim_cls
        question_file = './data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_origin_val.jsonl'
    elif args.dataset == 'rsna':
        dataset_cls = rsna_cls
        question_file = './data/rsna/rsna_pneumonia_llava_origin_val.jsonl'
    elif args.dataset == 'covid-cxr2':
        dataset_cls = covid_cls
        question_file = './data/COVIDx_CXR/COVIDx_CXR_llava_origin_val.jsonl'
  
        
    original_class.extend(item for item in dataset_cls if item not in original_class)
    # original_class = dataset_cls
    mapping = []
    for disease in dataset_cls:
        if disease in original_class:
            print(disease)
            mapping.append(original_class.index(disease))
        else:
            mapping.append(-1)
    MIMIC_mapping = [ _ for i,_ in enumerate(mapping) if _ != -1] # valid MIMIC class index
    dataset_mapping = [ i for i,_ in enumerate(mapping) if _ != -1] # valid (exist in MIMIC) chexray class index
    target_class = [dataset_cls[i] for i in dataset_mapping ] # Filter out non-existing class
    
    return target_class,question_file

def eval_model(args, classes,question_file):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map='cuda:0')

    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # questions = random.sample(questions, min(100, len(questions)))
    
    answers_file = os.path.expanduser(args.output_path)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["label"]
        image_file = os.path.join(args.image_folder,line["image"])
        
        qs = line["text"].replace('<image>', '').strip()
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        cur_prompt = '<image>' + '\n' + cur_prompt
        # qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
        # cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
        if args.use_cot:
            cot = " Let's think step by step."
        else:
            cot=""
        qs = qs + cot
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(0)

        try:
            if args.dataset == 'rsna':
                img = pydicom.dcmread(image_file).pixel_array  # 读取 DICOM 图像数据
                img = img.astype(float) / 255.0  # 归一化图像
                img = exposure.equalize_hist(img)  # 直方图均衡化

                # 转换为 PIL 图像并应用预处理
                img = (255 * img).astype(np.uint8)  # 转换为 uint8 类型
                image = Image.fromarray(img).convert('RGB') 
                # image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)[0].cuda(0)
            else:
                image = Image.open(image_file).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)[0].to(device)
        except Exception as e:
            print(f"Warning: Skipping image {image_file} due to error: {e}")
            continue 
        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(0),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    
def clip_eval_model(args,classes,question_file):
    # Model
    # disable_torch_init()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device_map='cuda:0'
    )
  
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
            return_dict=True
        )
        # 获取类别特征的最后一个隐藏层并计算均值

        global_category_embedding = category_output.hidden_states[-2].mean(dim=1)
        global_category_embeddings_cache.append(global_category_embedding)
        # local_category_embeddings_cache.append(local_category_embedding)
        
    # 将类别特征向量拼接成 (N, C) 的矩阵，其中N是类别数量，C是特征维度
    global_category_embeddings_cache = torch.cat(global_category_embeddings_cache, dim=0).to(device)
    # local_category_embeddings_cache = torch.cat(local_category_embeddings_cache, dim=0).to(device)
    # # 打印类别特征向量的维度
    print('Global Category embeddings:', global_category_embeddings_cache)   
    # print('Local Category embeddings:', local_category_embeddings_cache)          

    questions = [
        json.loads(q) for q in open(os.path.expanduser(question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # questions = random.sample(questions, min(100, len(questions)))

    # 存储真实标签和预测结果
    all_labels = []
    all_probs = []
    letter_to_disease = {chr(65 + idx): disease for idx, disease in enumerate(classes)}
    for line in tqdm(questions):
        img_path = args.image_folder + line["image"]
        qs = line["text"]
        label_dict = line["label"]
        
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
        
        # 尝试加载图像，如果遇到异常则跳过该图像
        try:
            if args.dataset == 'rsna':
                img = pydicom.dcmread(img_path).pixel_array  # 读取 DICOM 图像数据
                img = img.astype(float) / 255.0  # 归一化图像
                img = exposure.equalize_hist(img)  # 直方图均衡化

                # 转换为 PIL 图像并应用预处理
                img = (255 * img).astype(np.uint8)  # 转换为 uint8 类型
                image = Image.fromarray(img).convert('RGB') 
                # image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)[0].cuda(0)
            else:
                image = Image.open(img_path).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)[0].to(device)
        except Exception as e:
            print(f"Warning: Skipping image {img_path} due to error: {e}")
            continue 
        
        with torch.inference_mode():
            outputs = model.inference_pipeline(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                global_category_embeddings_cache=global_category_embeddings_cache,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                use_cache=True,
            )

            # similarity_probs 是一个 (batch_size, num_classes) 的矩阵
            similarity_probs = outputs  # 已经 softmax 过了，得到每个类别的预测概率
            # 获取真实标签
        
            true_labels = torch.zeros(len(classes))  # 假设 `classes` 是类别列表
            for disease, value in label_dict.items():
                if value == 1 and disease in classes:
                    true_labels[classes.index(disease)] = 1
                    
            # 将标签和预测概率存储到全局变量
            all_labels.append(true_labels.cpu().numpy())
            all_probs.append(similarity_probs.cpu().numpy())

    # 将 all_labels 和 all_probs 转换为 numpy 数组
    all_labels = np.array(all_labels)  # shape: (num_samples, num_classes)
    all_probs = np.array(all_probs).squeeze(1)  # shape: (num_samples, num_classes)

    result_metrics = {}

    # 计算每个类别的准确率、AUC、AUPRC、F1、精确度、召回率
    accuracies, auc_scores, auprc_scores, f1_scores, recall_scores, precision_scores = [], [], [], [], [], []
    
    for i in range(all_labels.shape[1]):
        # 计算精确度、召回率和阈值
        precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])

        # 计算 F1 分数并找到最大值
        f1 = 2 * precision * recall / (precision + recall + 1e-8)  # 避免分母为0
        max_f1_idx = np.argmax(f1)  # 最大 F1 对应的索引
        
        # 选择最大 F1 对应的阈值
        best_threshold = thresholds[max_f1_idx]
        
        # 二值化预测并计算准确率
        all_predictions_binary = (all_probs[:, i] >= best_threshold).astype(int)
        accuracy = (all_predictions_binary == all_labels[:, i]).mean()

        # 计算 AUC 和 AUPRC
        try:
            auc_score = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            # pass
            auc_score = np.nan  # 如果该类别标签全为0或1，返回 NaN
        
        # 计算 AUPRC
        auprc_score = auc(recall, precision)

        # 保存每个类别的指标
        accuracies.append(accuracy)
        auc_scores.append(auc_score)
        auprc_scores.append(auprc_score)
        f1_scores.append(np.max(f1))
        recall_scores.append(recall[max_f1_idx])
        precision_scores.append(precision[max_f1_idx])

    # 汇总结果
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
    print(f"\n===== Evaluation Metrics for {args.dataset} =====")
    for key, value in result_metrics.items():
        if isinstance(value, (list, np.ndarray)):  # 打印所有元素
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    # 创建结果目录
    result_dir = os.path.dirname(args.result_file)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    # 写入结果文件
    with open(args.result_file, 'w') as f:
        for key, value in result_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Results saved to {args.result_file}")



# 通过疾病名字计算指标
def get_metrics1(args,classes,question_file):
    # 读取数据

    answers = [json.loads(line) for line in open(args.output_path)]

    disease_list = classes
    
    print(f"Total number of answers: {len(answers)}")

    disease_to_idx = {disease: idx for idx, disease in enumerate(disease_list)}

    # 存储真实标签和预测标签
    y_true = []
    y_pred = []

    # 遍历每个 answer，提取 labels 和预测类别
    for item in answers:
        # 获取标签（label），可能包含多个标签
        labels = item["question_id"]

        # 获取预测的 text
        text = item["text"].lower()

        # 预测每个疾病是否在 text 中
        predicted_categories = [1 if disease in text else 0 for disease in disease_list]

        # 生成真实标签向量
        true_labels = torch.zeros(len(disease_list))  # 假设 `classes` 是类别列表
        for disease, value in labels.items():
            if value == 1 and disease in disease_list:
                true_labels[disease_list.index(disease)] = 1
        y_true.append(true_labels)
        y_pred.append(predicted_categories)

    # 转换为 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算 AUC（先筛选掉全 0 或全 1 的类别）
    valid_indices = [i for i in range(len(disease_list)) if len(set(y_true[:, i])) > 1]

    if valid_indices:
        auc_micro = roc_auc_score(y_true[:, valid_indices], y_pred[:, valid_indices], average='micro')
        auc_macro = roc_auc_score(y_true[:, valid_indices], y_pred[:, valid_indices], average='macro')
    else:
        auc_micro, auc_macro = 0, 0  # 避免计算错误

    # 计算 F1 分数
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # 计算每个类别的准确率
    category_accuracies = (y_true * y_pred).sum(axis=0) / y_true.sum(axis=0) * 100
    category_accuracies = {disease: acc if not np.isnan(acc) else 0 for disease, acc in zip(disease_list, category_accuracies)}

    # 计算类别平均准确率
    average_accuracy = sum(category_accuracies.values()) / len(category_accuracies)

    # 输出结果
    print(f"Category accuracies: {category_accuracies}")
    print(f"Average accuracy: {average_accuracy}%")
    print(f"AUC (Micro): {auc_micro}")
    print(f"AUC (Macro): {auc_macro}")
    print(f"F1 Score (Micro): {f1_micro}")
    print(f"F1 Score (Macro): {f1_macro}")

    result = {
        "category_accuracies": category_accuracies,
        "average_accuracy": average_accuracy,
        "auc_micro": auc_micro,
        "auc_macro": auc_macro,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro
    }

    result_dir = os.path.dirname(args.result_file)

    # 确保目录存在
    os.makedirs(result_dir, exist_ok=True)
    with open(args.result_file, 'w') as f:
        json.dump(result, f, indent=4)

# 通过疾病索引计算指标 A,B,C,D...

def get_metrics2(args,classes,question_file):
    # 读取数据
    answers = [json.loads(line) for line in open(args.output_path)]

    # 疾病类别及其索引 A, B, C, D...
    # disease_list = [
    #     'fibrosis', 'edema', 'pneumothorax', 'cardiomegaly', 'atelectasis', 
    #     'nodule', 'emphysema', 'no finding', 'mass', 'pleural_thickening', 
    #     'effusion', 'infiltration', 'pneumonia', 'hernia', 'consolidation'
    # ]
    disease_indices = [chr(65 + i) for i in range(len(classes))]  # A, B, C, ..., O

    print(f"Total number of answers: {len(answers)}")

    # 存储真实标签和预测标签
    y_true = []
    y_pred = []

    # 遍历每个 answer，提取 labels 和预测类别
    for item in answers:
        # 真实标签（A, B, C, D...）
        labels = item["question_id"]

        # 获取预测的 text
        text = item["text"].upper().strip()  # 转换为大写匹配 A, B, C, D...
        predicted_categories = [1 if disease in text else 0 for disease in disease_indices]

        # 生成真实标签向量
        true_labels = [1 if disease in labels else 0 for disease in disease_indices]

        y_true.append(true_labels)
        y_pred.append(predicted_categories)

    # 转换为 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算 AUC（先筛选掉全 0 或全 1 的类别）
    valid_indices = [i for i in range(len(disease_indices)) if len(set(y_true[:, i])) > 1]

    if valid_indices:
        auc_micro = roc_auc_score(y_true[:, valid_indices], y_pred[:, valid_indices], average='micro')
        auc_macro = roc_auc_score(y_true[:, valid_indices], y_pred[:, valid_indices], average='macro')
    else:
        auc_micro, auc_macro = 0, 0  # 避免计算错误

    # 计算 F1 分数
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # 计算每个类别的准确率
    category_accuracies = (y_true * y_pred).sum(axis=0) / y_true.sum(axis=0) * 100
    category_accuracies = {
        classes[i]: acc if not np.isnan(acc) else 0 
        for i, acc in enumerate(category_accuracies)
    }

    # 计算类别平均准确率
    average_accuracy = sum(category_accuracies.values()) / len(category_accuracies)

    # 输出结果
    print(f"Category accuracies: {category_accuracies}")
    print(f"Average accuracy: {average_accuracy}%")
    print(f"AUC (Micro): {auc_micro}")
    print(f"AUC (Macro): {auc_macro}")
    print(f"F1 Score (Micro): {f1_micro}")
    print(f"F1 Score (Macro): {f1_macro}")

    result = {
        "category_accuracies": category_accuracies,
        "average_accuracy": average_accuracy,
        "auc_micro": auc_micro,
        "auc_macro": auc_macro,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro
    }

    with open(args.result_file, 'w') as f:
        json.dump(result, f, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/srv/lby/llava_med/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--use-cot", type=int, default=0)
    parser.add_argument("--output-path", type=str, default="./data/chest_xray/Chest-X-ray_llava_origin_val_ans.jsonl")
    parser.add_argument("--dataset", type=str, default="siim")
    parser.add_argument("--chexpert-subset", type=str, default="False")
    parser.add_argument("--result-file", type=str, default="./result/chest_xray/Chest-X-ray_classify.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--inference", type=str, default="origin")
    args, remaining_args = parser.parse_known_args()
    
    # Use HfArgumentParser for SparseArguments
    # hf_parser = HfArgumentParser(SparseArguments)
    # sparse_args, = hf_parser.parse_args_into_dataclasses(remaining_args)
    
  
    # /srv/lby/llava_med/checkpoints/llava-mistral_new_clip_ft2
    # /srv/lby/llava_med/checkpoints/llava-llava-mistral_ft2
    

    classes,question_file = get_classes(args)
    if  args.inference == "clip":
        clip_eval_model(args,classes,question_file)
    else:
        eval_model(args,classes,question_file)
        get_metrics1(args,classes,question_file)
        
        # # 根据 model_path 选择合适的 metrics 计算方式
        # if "llava_med" in args.model_path.lower():
        #     get_metrics1(args,classes,question_file )
        # else:
        #     if "sft" in args.model_path.lower():
        #         get_metrics1(args,classes,question_file)
        #     else:
        #         get_metrics2(args,classes,question_file)  