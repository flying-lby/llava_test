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

import pydicom
from skimage import exposure

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
    Book_choice: int = 1


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



def test(args, sparse_args):
    # Model
    # disable_torch_init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------- 模型加载与全局类别嵌入计算 ---------------------
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, sparse_args, device_map=None  # 让 DataParallel 处理 device
    )

    # 多 GPU 推理
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])  # 指定 GPU 0 和 1
    model.to(device)
    model.eval()
    if hasattr(model, "module"):
        model_config = model.module.config
    else:
        model_config = model.config
    
    # 加载类别数据
    chexray14_cls = [ 'atelectasis', 'cardiomegaly', 'effusion', 'infiltrate', 'mass', 'nodule', 'pneumonia',
                    'pneumothorax', 'consolidation', 'edema', 'emphysema', 'fibrosis', 'thicken', 'hernia']  #Fibrosis seldom appears in MIMIC_CXR and is divided into the 'tail_abnorm_obs' entitiy.  
    mura_cls = lera_cls = ['abnormality']
    if args.dataset == 'chexpert':
        chexpert_subset = args.chexpert_subset

        if chexpert_subset == 'False':
            chexpert_cls = ['normal', 'enlarge', 'cardiomegaly',
                'opacity', 'lesion', 'edema', 'consolidation', 'pneumonia', 'atelectasis',
                'pneumothorax', 'effusion', "abnormality", 'fracture', 'device']
        else:
            chexpert_cls = ['cardiomegaly','edema', 'consolidation', 'atelectasis','effusion']

    siim_cls = ['pneumothorax', 'non-pneumothorax']
    rsna_cls = ['pneumonia','normal']
    covid_cls = ['covid19']

    padchest_seen_class = ['normal', 'pleural effusion', 'pacemaker', 'atelectasis', 'pneumonia', 'consolidation', 'cardiomegaly', 'emphysema', 
                           'nodule', 'edema', 'pneumothorax', 'fracture', 'mass', 'catheter']


    original_class = [
                'normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process', 'abnormality', 'enlarge', 'tip', 'low',
                'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
                'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
                'tail_abnorm_obs', 'excluded_obs'
            ]
    
    padchest_rare = ['suture material', 'sternotomy', 'supra aortic elongation', 'metal', 'abnormal foreign body', 'central venous catheter via jugular vein', 'vertebral anterior compression', 'diaphragmatic eventration', #'consolidation', 
    'calcified densities', 'volume loss', 'single chamber device', 'vertebral compression', 'bullas', 'axial hyperostosis', 'aortic button enlargement', 'calcified granuloma', 'clavicle fracture', 'dual chamber device', 'mediastinic lipomatosis',
                     'esophagic dilatation', 'azygoesophageal recess shift', 'breast mass', 'round atelectasis', 'surgery humeral', 'aortic aneurysm', 'nephrostomy tube', 'sternoclavicular junction hypertrophy', 'pulmonary artery hypertension', 'pleural mass', 'empyema', 'external foreign body', 'respiratory distress', 'total atelectasis', 'ventriculoperitoneal drain tube', 'right sided aortic arch', 'aortic endoprosthesis', 'cyst', 'pulmonary venous hypertension', 'double J stent']
    
    padchest_unseen_class = [
        'hypoexpansion basal', 'non axial articular degenerative changes', 'central venous catheter via jugular vein', 'multiple nodules', 
        'COPD signs', 'calcified densities', 'mediastinal shift', 'hiatal hernia', 
        'volume loss', 'mediastinic lipomatosis', 'central venous catheter', 
        'ground glass pattern', 'surgery lung', 'miliary opacities', 'sclerotic bone lesion', 'pleural plaques', 'osteosynthesis material', 
        'calcified mediastinal adenopathy', 'apical pleural thickening', 'aortic elongation', 'major fissure thickening', 'callus rib fracture', 
        'pulmonary venous hypertension', 'cervical rib', 'loculated pleural effusion', 
        'flattened diaphragm' 
    ]

    padchest_unseen_class = list(set(padchest_unseen_class + padchest_rare))
    if args.dataset == 'chestxray':
        dataset_cls = chexray14_cls
        question_file = './data/chest_xray/chest_xray_llava_val.jsonl'
        result_file = args.result_folder + 'Chest_Xray_classify.txt'
    elif args.dataset == 'chexpert':
        dataset_cls = chexpert_cls
        question_file = './data/chexpert/chexpert_llava_val.jsonl'
        result_file = args.result_folder + 'chexpert_classify.txt'
    elif args.dataset == 'siim':
        dataset_cls = siim_cls
        question_file = './data/SIIM_Pneumothorax/SIIM_Pneumothorax_llava_val.jsonl'
        result_file = args.result_folder + 'siim_classify.txt'
    elif args.dataset == 'rsna':
        dataset_cls = rsna_cls
        question_file = './data/rsna/rsna_pneumonia_llava.jsonl'
        result_file = args.result_folder + 'rsna_classify.txt'
    elif args.dataset == 'covid-cxr2':
        dataset_cls = covid_cls
        original_class.append('covid19')
    elif args.dataset == 'covid-r':
        dataset_cls = covid_cls
        original_class.append('covid19')
    elif args.dataset == 'padchest':
        # dataset_cls = padchest_seen_class + padchest_unseen_class
        if args.subdata == 'unseen':
            original_class += padchest_unseen_class
            dataset_cls = padchest_unseen_class
            result_file = args.result_folder + 'padchest_unseen_classify.txt'
        elif args.subdata == 'rare':
            dataset_cls = padchest_rare
            result_file = args.result_folder + 'padchest_rare_classify.txt'
        else:
            dataset_cls = padchest_seen_class
            result_file = args.result_folder + 'padchest_seen_classify.txt'
        question_file = './data/padchest/padchest_llava_val.jsonl'
        
        # if 'pleural effusion' in dataset_cls:
        #     dataset_cls[dataset_cls.index('pleural effusion')] = 'effusion'
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
    # target_class = rsna_cls
    print(MIMIC_mapping)


    # 确保类别数据是一个列表
    categories = [f"This is a chest X-ray showing {category}" for category in target_class]

    # 对类别进行编码
    encoded_categories = [tokenizer(category, return_tensors="pt") for category in categories]
    category_ids = pad_sequence([item.input_ids.squeeze(0) for item in encoded_categories], batch_first=True).to(device)
    category_attention_mask = pad_sequence([item.attention_mask.squeeze(0) for item in encoded_categories], batch_first=True).to(device)
    
    # 类别特征向量存储, 只需要计算一次
    global_category_embeddings_cache = []
    sparse_args_dict = asdict(sparse_args)
    with torch.no_grad():
        for i in range(category_ids.size(0)):
            category_input_ids = category_ids[i].unsqueeze(0).to(device)
            category_attention = category_attention_mask[i].unsqueeze(0).to(device)

            category_output = model(
                input_ids=category_input_ids, 
                attention_mask=category_attention,
                output_hidden_states=True,
                return_emb=True,
                return_dict=True
            )

            # 取最后指定层的隐藏状态，并取末尾 Txtcls_count 个 token
            global_category_embedding = category_output.hidden_states[-sparse_args_dict["feature_layer"]][:, -sparse_args_dict["Txtcls_count"]:]
            global_category_embedding = model.module.txt_mlp(global_category_embedding) if hasattr(model, 'module') else model.txt_mlp(global_category_embedding)
            global_category_embedding = global_category_embedding.mean(dim=1)
            global_category_embeddings_cache.append(global_category_embedding)
    
    global_category_embeddings_cache = torch.cat(global_category_embeddings_cache, dim=0).to(device)
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

    for line in tqdm(questions):
        img_path = args.image_folder + line["image"]
        qs = line["text"]
        label_dict = line["label"]
        
        if model_config.mm_use_im_start_end:
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
                image_tensor = process_images([image], image_processor, model_config)[0].cuda(0)
            else:
                image = Image.open(img_path).convert("RGB")
                image_tensor = process_images([image], image_processor, model_config)[0].to(device)
        except Exception as e:
            print(f"Warning: Skipping image {img_path} due to error: {e}")
            continue 
        
        with torch.inference_mode():
            outputs = model.module.inference_pipeline(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                global_category_embeddings_cache=global_category_embeddings_cache,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                use_cache=True,
            ) if hasattr(model, 'module') else model.inference_pipeline(
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
        
            true_labels = torch.zeros(len(target_class))  # 假设 `classes` 是类别列表
            for disease, value in label_dict.items():
                if value == 1 and disease in target_class:
                    true_labels[target_class.index(disease)] = 1
           
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
    result_dir = os.path.dirname(result_file)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    # 写入结果文件
    with open(result_file, 'w') as f:
        for key, value in result_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Results saved to {result_file}")

# def eval_model_chest_xray(args, sparse_args):
#     # Model
#     # disable_torch_init()
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model_path = os.path.expanduser(args.model_path)
#     model_name = get_model_name_from_path(model_path)

#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path, args.model_base, model_name, sparse_args, device_map='cuda:0'
#     )
    
#     # 加载类别数据
#     with open('./data/chest_xray/Chest-X-ray_classes.json', 'r') as f:
#         classes = json.load(f)

#     # 确保类别数据是一个列表
#     categories = [f"This is a chest X-ray showing {category}" for category in classes]

#     # 对类别进行编码
#     encoded_categories = [tokenizer(category, return_tensors="pt") for category in categories]
#     category_ids = pad_sequence([item.input_ids.squeeze(0) for item in encoded_categories], batch_first=True).to(device)
#     category_attention_mask = pad_sequence([item.attention_mask.squeeze(0) for item in encoded_categories], batch_first=True).to(device)
    
#     # 类别特征向量存储, 只需要计算一次
#     global_category_embeddings_cache = []
#     local_category_embeddings_cache = []
#     for i in range(category_ids.size(0)):
#         category_input_ids = category_ids[i].unsqueeze(0)
#         category_attention = category_attention_mask[i].unsqueeze(0)
    
#         # 获取每个类别的特征向量
#         category_output = model.forward(
#             input_ids=category_input_ids, 
#             attention_mask=category_attention,
#             output_hidden_states=True,
#             return_emb = True,
#             return_dict=True
#         )
#         # 获取类别特征的最后一个隐藏层并计算均值
#         sparse_args_dict = asdict(sparse_args)
#         global_category_embedding = category_output.hidden_states[-sparse_args_dict["feature_layer"]][:, -sparse_args_dict["Txtcls_count"]:]
#         # local_category_embedding = category_output.hidden_states[-sparse_args_dict["feature_layer"]][:, :-sparse_args_dict["ncls_count"]].mean(dim=1)
        
#         global_category_embedding = model.txt_mlp(global_category_embedding)
#         global_category_embedding = global_category_embedding.mean(dim=1)
#         # local_category_embedding = model.mis_mlp(local_category_embedding)
        
#         global_category_embeddings_cache.append(global_category_embedding)
#         # local_category_embeddings_cache.append(local_category_embedding)
        
#     # 将类别特征向量拼接成 (N, C) 的矩阵，其中N是类别数量，C是特征维度
#     global_category_embeddings_cache = torch.cat(global_category_embeddings_cache, dim=0).to(device)
#     # local_category_embeddings_cache = torch.cat(local_category_embeddings_cache, dim=0).to(device)
#     # # 打印类别特征向量的维度
#     print('Global Category embeddings:', global_category_embeddings_cache)   
#     # print('Local Category embeddings:', local_category_embeddings_cache)          

#     questions = [
#         json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
#     ]
#     questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
#     questions = random.sample(questions, min(1000, len(questions)))

#     # 存储真实标签和预测结果
#     all_labels = []  # 真实标签
#     all_predictions = []  # 预测标签
#     all_probs = []  # 存储类别概率，用于计算 AUC

#     for line in tqdm(questions):
#         idx = line["question_id"]
#         image_file = line["image"]
#         qs = line["text"]
        
#         # 创建提示语句
#         cur_prompt = qs
#         if model.config.mm_use_im_start_end:
#             qs = (
#                 DEFAULT_IM_START_TOKEN
#                 + DEFAULT_IMAGE_TOKEN
#                 + DEFAULT_IM_END_TOKEN
#                 + "\n"
#                 + qs
#             )
#         else:
#             qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

#         conv = conv_templates[args.conv_mode].copy()
#         conv.append_message(conv.roles[0], qs)
#         conv.append_message(conv.roles[1], None)
#         prompt = conv.get_prompt()

#         input_ids = (
#             tokenizer_image_token(
#                 prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
#             )
#             .unsqueeze(0)
#             .cuda(0)
#         )
#         attention_mask = (input_ids != tokenizer.pad_token_id).long().cuda(0)
        
#         image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
#         image_tensor = process_images([image], image_processor, model.config)[0].cuda(0)
        
#         with torch.inference_mode():
#             outputs = model.inference_pipeline(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask, 
#                 global_category_embeddings_cache=global_category_embeddings_cache,
#                 # local_category_embeddings_cache=local_category_embeddings_cache,
#                 images=image_tensor.unsqueeze(0).half().cuda(0),
#                 image_sizes=[image.size],
#                 use_cache=True,
#             )

#             # similarity_probs 是一个 (batch_size, num_classes) 的矩阵
#             similarity_probs = outputs  # 已经 softmax 过了，得到每个类别的预测概率
#             # 获取真实标签
#             labels = ["-".join(line["question_id"].split("-")[1:])]  # 获取标签
#             labels = [label.lower() for label in labels]  # 转为小写以方便比较

#             # 计算该样本的真实标签
#             true_labels = torch.zeros(len(classes))  # 假设 `classes` 是类别列表
#             for label in labels:
#                 # 将逗号分隔的多标签拆分为单个类别
#                 split_labels = [lbl.strip() for lbl in label.split(",")]
#                 for split_label in split_labels:
#                     if split_label in classes:
#                         # 在这里设置真实标签为 1，如果该样本属于该类别
#                         true_labels[classes.index(split_label)] = 1
#                     else:
#                         print(f"Warning: {split_label} not found in classes.")

#             # 将标签和预测概率存储到全局变量
#             all_labels.append(true_labels.cpu().numpy())
#             all_probs.append(similarity_probs.cpu().numpy())

#     # 将 all_labels 和 all_probs 转换为 numpy 数组
#     all_labels = np.array(all_labels)  # shape: (num_samples, num_classes)
#     all_probs = np.array(all_probs).squeeze(1)  # shape: (num_samples, num_classes)

#     # 初始化准确率列表
#     accuracies = []
#     result_metrics = {}

#     # 计算每个类别的准确率
#     for i in range(all_labels.shape[1]):
#         # 获取当前类别的精确度、召回率和阈值
#         precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        
#         # 计算 F1 分数
#         f1 = 2 * precision * recall / (precision + recall + 1e-8)  # 避免分母为0
#         max_f1_idx = np.argmax(f1)  # 取最大 F1 分数对应的索引
        
#         # 选择最大 F1 分数时的阈值
#         best_threshold = thresholds[max_f1_idx]
        
#         # 根据该阈值对预测值进行二值化（预测为1的样本）
#         all_predictions_binary = (all_probs[:, i] >= best_threshold).astype(int)
        
#         # 计算该类别的准确率
#         accuracy = (all_predictions_binary == all_labels[:, i]).mean()
#         accuracies.append(accuracy)

#     # 计算 AUC（逐类计算 AUC）
#     auc_scores = []
#     for i in range(all_labels.shape[1]):  # 对每个类别计算 AUC
#         try:
#             auc_score = roc_auc_score(all_labels[:, i], all_probs[:, i])
#             auc_scores.append(auc_score)
#         except ValueError:
#             # 如果该类别的标签都为0或1，roc_auc_score会抛出 ValueError
#             auc_scores.append(np.nan)

#     # 计算 AUPRC（逐类计算 AUPRC）
#     auprc_scores = []
#     for i in range(all_labels.shape[1]):  # 对每个类别计算 AUPRC
#         precision, recall, _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
#         auprc_score = auc(recall, precision)  # 计算 AUPRC
#         auprc_scores.append(auprc_score)

#     # 计算每个类别的精确度、召回率和 F1 分数
#     f1_scores = []
#     recall_scores = []
#     precision_scores = []
#     for i in range(all_labels.shape[1]):
#         # 计算每个类别的精确度、召回率和 F1 分数
#         precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])

#         # 计算 F1 分数
#         f1 = 2 * precision * recall / (precision + recall + 1e-8)  # 避免分母为0
#         max_f1 = np.max(f1)  # 取最大 F1 分数
#         f1_scores.append(max_f1)

#         # 记录召回率（对应最大 F1 的召回率）
#         recall_scores.append(recall[np.argmax(f1)])

#         # 记录精确度
#         precision_scores.append(precision[np.argmax(f1)])
        
        
#     result_metrics["mean_accuracy"] = np.mean(accuracies)
#     result_metrics["mean_auc"] = np.nanmean(auc_scores)
#     result_metrics["mean_f1"] = np.mean(f1_scores)
#     result_metrics["mean_auprc"] = np.mean(auprc_scores)
#     result_metrics["mean_recall"] = np.mean(recall_scores)
#     result_metrics["mean_precision"] = np.mean(precision_scores)
    
#     result_metrics["accuracies_per_class"] = accuracies
#     result_metrics["auc_scores_per_class"] = auc_scores
#     result_metrics["auprc_scores_per_class"] = auprc_scores
#     result_metrics["f1_scores_per_class"] = f1_scores
#     result_metrics["recall_scores_per_class"] = recall_scores
#     result_metrics["precision_scores_per_class"] = precision_scores

#     # 打印所有计算的结果
#     print("\n===== Evaluation Metrics =====")
#     for key, value in result_metrics.items():
#         if isinstance(value, list) or isinstance(value, np.ndarray):  # 打印所有元素
#             print(f"{key}: {value}")
#         else:
#             print(f"{key}: {value}")
    
#     # 检查目录并创建
#     result_dir = os.path.dirname(args.result_file)  # 提取文件路径的目录部分
#     if result_dir and not os.path.exists(result_dir):  # 如果目录不存在
#         os.makedirs(result_dir, exist_ok=True)  # 创建目录

#     # 写入文件
#     with open(args.result_file, 'w') as f:
#         for key, value in result_metrics.items():
#             f.write(f"{key}: {value}\n")

#     print(f"Results saved to {args.result_file}")


def eval_model_SIIM(args, sparse_args):
    # Setup device and load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, sparse_args, device_map='cuda:0'
    )

    # 二分类：气胸（pneumothorax）与正常（normal）
    categories = ["This is a chest X-ray showing normal.", "This is a chest X-ray showing pneumothorax."]

    # 对类别进行编码
    encoded_categories = tokenizer(categories, padding=True, return_tensors="pt").to(device)
    category_ids = encoded_categories['input_ids']
    category_attention_mask = encoded_categories['attention_mask']

    # 获取气胸和正常类别的特征向量
    global_category_embeddings_cache = []
    for i in range(category_ids.size(0)):
        category_input_ids = category_ids[i].unsqueeze(0)
        category_attention = category_attention_mask[i].unsqueeze(0)

        # 获取类别特征
        category_output = model.forward(
            input_ids=category_input_ids, 
            attention_mask=category_attention,
            output_hidden_states=True,
            return_emb=True,
            return_dict=True
        )
        # 提取类别特征的最后一个隐藏层并计算均值
        sparse_args_dict = asdict(sparse_args)
        global_category_embedding = category_output.hidden_states[-sparse_args_dict["feature_layer"]][:, -sparse_args_dict["Txtcls_count"]:]
        global_category_embedding = model.txt_mlp(global_category_embedding)
        global_category_embedding = global_category_embedding.mean(dim=1)

        global_category_embeddings_cache.append(global_category_embedding)

    # 将类别特征向量拼接成 (2, C) 的矩阵，包含气胸和正常两个类别的特征
    global_category_embeddings_cache = torch.cat(global_category_embeddings_cache, dim=0).to(device)

    # 加载问题数据
    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # 存储真实标签和预测结果
    all_labels = []  # 真实标签
    all_probs = []  # 存储类别的预测概率，用于计算 AUC

    # 对每个问题进行推理
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]

        # 创建提示语句
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        input_ids = (
            tokenizer_image_token(
                qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
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
                images=image_tensor.unsqueeze(0).half().cuda(0),
                image_sizes=[image.size],
                use_cache=True,
            )

        # similarity_probs 是一个 (batch_size, 2) 的矩阵，表示气胸和正常两个类别的概率
        similarity_probs = outputs  # 已经 softmax 过了，得到每个类别的预测概率

        # 对于二分类任务，获取气胸类别（pneumothorax）的概率
        probs = similarity_probs[0, 1].item()  # 获取气胸类别的概率（索引1）

        # 获取真实标签，气胸为 1，正常为 0
        true_labels = torch.zeros(1)
        true_labels[0] = 1 if 'pneumothorax' in line["question_id"].lower() else 0

        # 将标签和预测概率存储到全局变量
        all_labels.append(true_labels.cpu().numpy())  # 保存真实标签
        all_probs.append(probs)  # 保存预测概率

    # 将 all_labels 和 all_probs 转换为 numpy 数组
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs).squeeze()  # 去除多余的维度，确保是一维数组

    # 计算每个类别的 precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)  # 避免分母为0

    # 选择最大 F1 分数所对应的最佳阈值
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]

    # 使用最佳阈值对预测结果进行二值化
    binary_predictions = (all_probs >= best_threshold).astype(int)

    # 计算准确率（Accuracy）
    accuracy = (binary_predictions == all_labels).mean()

    # 计算 AUC 和 F1 分数
    auc_score = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, binary_predictions)  # 使用最佳阈值二值化后的预测值计算 F1 分数

    # 计算 AUPRC
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    auprc_score = auc(recall, precision)

    # 打印评估指标
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc_score}")
    print(f"AUPRC: {auprc_score}")
    print(f"F1 Score: {f1}")
    print(f"Best Threshold: {best_threshold}")

    # 保存评估结果
    result_metrics = {
        "accuracy": accuracy,
        "auc": auc_score,
        "auprc": auprc_score,
        "f1": f1,  # 添加 F1 分数
        "best_threshold": best_threshold,
    }

    # 检查目录并创建
    result_dir = os.path.dirname(args.result_file)  
    if result_dir and not os.path.exists(result_dir):  
        os.makedirs(result_dir, exist_ok=True)  

    # 写入文件
    with open(args.result_file, 'w') as f:
        for key, value in result_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Results saved to {args.result_file}")

def eval_model_chexpert(args, sparse_args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------- 模型加载与全局类别嵌入计算 ---------------------
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, sparse_args, device_map=None  # 让 DataParallel 处理 device
    )

    # 多 GPU 推理
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])  # 指定 GPU 0 和 1
    model.to(device)
    model.eval()
    if hasattr(model, "module"):
        model_config = model.module.config
    else:
        model_config = model.config

    # 加载类别数据
    # desc = [
    #     "No Finding - No abnormalities detected in the medical image. Characteristic: Normal appearance with no signs of pathology. Distinguishing Feature: Absence of any abnormal findings.",
        
    #     "Enlarged Cardiomediastinum - Widening of the central thoracic structures, possibly indicating mediastinal masses or vascular abnormalities. Characteristic: A prominent shadow in the center of the chest. Distinguishing Feature: Increased width of the mediastinum, often seen in the presence of large masses or aortic aneurysms.",
        
    #     "Cardiomegaly - Abnormal enlargement of the heart, often linked to heart failure, pericardial effusion, or hypertension. Characteristic: Enlarged cardiac silhouette on chest radiograph. Distinguishing Feature: Heart size exceeding the normal thoracic ratio, often associated with congestive heart failure or systemic hypertension.",
        
    #     "Lung Opacity - Increased lung density due to infections, tumors, fibrosis, or inflammatory processes. Characteristic: Focal or diffuse areas of opacity on imaging, indicating consolidation or infiltrates. Distinguishing Feature: Homogeneous or patchy density increase, often suggesting infections or tumors.",
        
    #     "Lung Lesion - A localized lung abnormality, which may be benign (e.g., granuloma) or malignant (e.g., lung cancer). Characteristic: Well-defined, often round or irregular mass or nodule. Distinguishing Feature: A lesion with irregular borders, potential growth pattern, or presence of calcifications, which can differentiate between benign and malignant causes.",
        
    #     "Edema - Fluid accumulation in lung tissues, commonly associated with heart failure, renal disease, or ARDS. Characteristic: Bilateral, diffuse interstitial and alveolar opacities in the lungs. Distinguishing Feature: Typically symmetric, with a 'bat wing' pattern around the hilum, and may show Kerley B lines indicative of pulmonary congestion.",
        
    #     "Consolidation - Alveolar filling with fluid, pus, blood, or cells, often seen in bacterial pneumonia or lung infections. Characteristic: Dense, homogeneous opacity filling a portion of the lung, often associated with air bronchograms. Distinguishing Feature: Alveolar opacities with air bronchograms, indicating pneumonia or other types of infection.",
        
    #     "Pneumonia - Infection-induced lung inflammation, typically presenting with fever, cough, and lung infiltrates on imaging. Characteristic: Consolidation, lobar or patchy opacity with a possible air bronchogram. Distinguishing Feature: Often localized, with specific lobe involvement, and may present with infiltrates and pleural effusions.",
        
    #     "Atelectasis - Partial or complete lung collapse due to airway obstruction, compression, or postoperative changes. Characteristic: Loss of lung volume, often with mediastinal shift toward the affected side. Distinguishing Feature: Focal or total lung collapse with associated displacement of structures like the heart and diaphragm.",
        
    #     "Pneumothorax - Presence of air in the pleural cavity, causing lung collapse and respiratory distress. Characteristic: Radiolucent area without lung markings, often seen at the apex. Distinguishing Feature: A clear, well-defined pleural line with absence of lung markings beyond it, often associated with a shift of the mediastinum.",
        
    #     "Pleural Effusion - Excess fluid buildup in the pleural space, which may result from infections, malignancies, or heart failure. Characteristic: Blunting of the costophrenic angles, and possibly a meniscus sign. Distinguishing Feature: Fluid accumulation in the pleural space, often causing the diaphragm and heart to appear displaced; commonly seen in heart failure or cancers.",
        
    #     "Pleural Other - Other pleural abnormalities, including thickening, calcifications, or pleural-based masses. Characteristic: Pleural-based lesions, thickening, or calcifications. Distinguishing Feature: Presence of pleural calcifications or thickening, typically seen in tuberculosis, asbestos-related diseases, or prior surgeries.",
        
    #     "Fracture - Breaks in the ribs, sternum, or clavicles, often due to trauma, osteoporosis, or pathological conditions. Characteristic: Radiolucent lines through the bones, often with adjacent soft tissue changes. Distinguishing Feature: Clear fracture lines in the bones, often with accompanying soft tissue swelling or hematoma, and commonly related to trauma or pathological conditions.",
        
    #     "Support Devices - Presence of medical devices such as catheters, pacemakers, chest tubes, or endotracheal tubes. Characteristic: Metallic or plastic devices with clearly identifiable shapes, such as tubes or wires. Distinguishing Feature: Foreign objects with clear identification, such as pacemakers, ventilator tubes, or drainage catheters, often placed for medical intervention."
    # ]
 
    # desc = [
    # "No Finding, No finding refers to the absence of radiographic abnormalities detected in the chest X-ray.", 
    # "Enlarged Cardiomediastinum", 
    # "Cardiomegaly, Cardiomegaly refers to the enlargement of the heart due to hypertension, cardiomyopathy, or valvular disease, causing chamber dilation or wall thickening. Imaging shows significant cardiac enlargement with an expanded and smooth contour, often marked by an increased cardiothoracic ratio, potentially accompanied by pulmonary congestion and bronchial congestion. Clinically, patients may experience reduced exercise tolerance, dyspnea, lower limb edema, and arrhythmias.", 
    # "Lung Opacity", 
    # "Lung Lesion", 
    # "Edema, Pulmonary edema refers to the abnormal accumulation of fluid in the pulmonary interstitium and alveoli, usually caused by cardiogenic or non-cardiogenic factors. Imaging shows patchy or 'bat-wing' distributed heterogeneous high-density shadows in the middle or entire lung, often accompanied by Kerley lines and cardiac enlargement. Clinically, patients typically experience acute dyspnea, cough, cyanosis, and bilateral lung crackles.", 
    # "Consolidation, Consolidation refers to the complete filling of alveolar spaces with liquid, pus, blood, or cellular material, replacing the normal air content. Imaging shows homogenous, dense, well-defined opacities, often with air bronchograms and pleural reactions, sometimes with minimal pleural effusion. Clinically, patients often have fever, cough, sputum production, chest pain, and dyspnea, with significantly elevated inflammatory markers.", 
    # "Pneumonia, Pneumonia refers to lung parenchyma inflammation caused by bacteria, viruses, fungi, or other microorganisms, leading to alveolar filling with inflammatory exudates. Imaging shows localized or patchy consolidation with irregular margins, often accompanied by air bronchograms, pleural reaction, and mild pleural effusion. Clinically, patients present with fever, cough, sputum production, chest pain, and fatigue, with elevated white blood cell counts and inflammatory markers.", 
    # "Atelectasis, Atelectasis refers to the collapse of part or all of the lung tissue due to airway obstruction, external thoracic pressure, or intrapulmonary pathology. Imaging shows increased local lung density, volume reduction, bronchial displacement, and visceral pleural traction, commonly affecting the lower lobes. Clinically, patients may exhibit rapid shallow breathing, localized decreased or absent breath sounds, and a history of recent surgery or inadequate airway clearance.", 
    # "Pneumothorax, Pneumothorax refers to the presence of air in the pleural cavity, leading to partial or complete lung collapse. Imaging typically shows a low-density black air space along the pleura, with a clear demarcation from the normal lung tissue, along with lung collapse. In tension pneumothorax, mediastinal shift may occur. Clinically, patients often present with sudden unilateral chest pain, dyspnea, and decreased breath sounds, sometimes accompanied by subcutaneous emphysema.", 
    # "Pleural Effusion, Pleural effusion refers to the abnormal accumulation of fluid in the pleural cavity, which may be caused by infection, heart failure, malignancy, or other inflammatory diseases. Typically seen in the lower lung fields and posterior chest cavity, imaging shows a homogeneous or layered fluid density with a clear meniscus sign, with CT revealing low-density regions. Severe effusion may cause lung compression or bronchial displacement. Clinically, patients may present with dyspnea, chest pain, and cough, with physical signs of reduced breath sounds, dull percussion, and abnormal auscultation.", 
    # "Pleural Other, Pleural thickening refers to fibrotic or calcified pleural changes due to chronic inflammation, infection, or asbestos exposure. Imaging shows localized or diffuse thickening along the pleural surface, appearing as streaky or patchy high-density shadows, sometimes with nodular changes. Clinically, patients may be asymptomatic, but a history of pleuritis or exposure to harmful substances is often present.", 
    # "Fracture", 
    # "Support Devices"
    # ]
    classes = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    ]

 
    categories = [f"This is a chest X-ray showing {category}" for category in classes]

    # 编码类别文本
    encoded_categories = [tokenizer(category, return_tensors="pt") for category in categories]
    category_ids = pad_sequence(
        [item.input_ids.squeeze(0) for item in encoded_categories],
        batch_first=True
    ).to(device)
    category_attention_mask = pad_sequence(
        [item.attention_mask.squeeze(0) for item in encoded_categories],
        batch_first=True
    ).to(device)

    # 计算全局类别特征向量（只计算一次）
    global_category_embeddings_cache = []
    sparse_args_dict = asdict(sparse_args)
    with torch.no_grad():
        for i in range(category_ids.size(0)):
            category_input_ids = category_ids[i].unsqueeze(0).to(device)
            category_attention = category_attention_mask[i].unsqueeze(0).to(device)

            category_output = model(
                input_ids=category_input_ids, 
                attention_mask=category_attention,
                output_hidden_states=True,
                return_emb=True,
                return_dict=True
            )

            # 取最后指定层的隐藏状态，并取末尾 Txtcls_count 个 token
            global_category_embedding = category_output.hidden_states[-sparse_args_dict["feature_layer"]][:, -sparse_args_dict["Txtcls_count"]:]
            global_category_embedding = model.module.txt_mlp(global_category_embedding) if hasattr(model, 'module') else model.txt_mlp(global_category_embedding)
            global_category_embedding = global_category_embedding.mean(dim=1)
            global_category_embeddings_cache.append(global_category_embedding)
    
    global_category_embeddings_cache = torch.cat(global_category_embeddings_cache, dim=0).to(device)

    # --------------------- 数据准备 ---------------------
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    all_labels = []
    all_probs = []

    # --------------------- 推理与指标计算 ---------------------
    for line in tqdm(questions):
        image_file = line["image"]
        qs = line["text"]

        if model_config.mm_use_im_start_end:
            qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs)
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        img_path = os.path.join(args.image_folder, image_file)
        image = Image.open(img_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model_config)[0].to(device)

        # **关键修改：使用多 GPU 进行推理**
        with torch.inference_mode():
            outputs = model.module.inference_pipeline(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                global_category_embeddings_cache=global_category_embeddings_cache,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                use_cache=True,
            ) if hasattr(model, 'module') else model.inference_pipeline(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                global_category_embeddings_cache=global_category_embeddings_cache,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                use_cache=True,
            )

        similarity_probs = outputs

        true_labels = torch.zeros(len(classes))
        label_dict = line["label"]
        for disease, value in label_dict.items():
            if value == 1 and disease in classes:
                true_labels[classes.index(disease)] = 1

        all_labels.append(true_labels.cpu().numpy())
        all_probs.append(similarity_probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs).squeeze(1)

    # --------------------- 计算性能指标 ---------------------
    accuracies, auc_scores, auprc_scores, f1_scores, precision_scores, recall_scores = [], [], [], [], [], []

    for i in range(all_labels.shape[1]):
        precision_vals, recall_vals, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        f1 = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-8)
        max_f1_idx = np.argmax(f1)
        best_threshold = thresholds[max_f1_idx]
        predictions_binary = (all_probs[:, i] >= best_threshold).astype(int)
        accuracies.append((predictions_binary == all_labels[:, i]).mean())

        try:
            auc_scores.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
        except ValueError:
            auc_scores.append(np.nan)

        auprc_scores.append(auc(recall_vals, precision_vals))
        f1_scores.append(np.max(f1))
        precision_scores.append(precision_vals[max_f1_idx])
        recall_scores.append(recall_vals[max_f1_idx])

    result_metrics = {
        "mean_accuracy": np.mean(accuracies),
        "mean_auc": np.nanmean(auc_scores),
        "mean_f1": np.mean(f1_scores),
        "mean_auprc": np.mean(auprc_scores),
        "mean_precision": np.mean(precision_scores),
        "mean_recall": np.mean(recall_scores),
    }

    print("\n===== Evaluation Metrics =====")
    for key, value in result_metrics.items():
        print(f"{key}: {value}")

    result_dir = os.path.dirname(args.result_file)
    os.makedirs(result_dir, exist_ok=True) if result_dir and not os.path.exists(result_dir) else None

    with open(args.result_file, 'w') as f:
        for key, value in result_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Results saved to {args.result_file}")
    
    
def eval_model_rsna(args, sparse_args):
    # Setup device and load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, sparse_args, device_map='cuda:0'
    )

    # 二分类：气胸（pneumothorax）与正常（normal）
    categories = ["This is a chest X-ray showing normal.", "This is a chest X-ray showing pneumonia."]

    # 对类别进行编码
    encoded_categories = tokenizer(categories, padding=True, return_tensors="pt").to(device)
    category_ids = encoded_categories['input_ids']
    category_attention_mask = encoded_categories['attention_mask']

    # 获取气胸和正常类别的特征向量
    global_category_embeddings_cache = []
    for i in range(category_ids.size(0)):
        category_input_ids = category_ids[i].unsqueeze(0)
        category_attention = category_attention_mask[i].unsqueeze(0)

        # 获取类别特征
        category_output = model.forward(
            input_ids=category_input_ids, 
            attention_mask=category_attention,
            output_hidden_states=True,
            return_emb=True,
            return_dict=True
        )
        # 提取类别特征的最后一个隐藏层并计算均值
        sparse_args_dict = asdict(sparse_args)
        global_category_embedding = category_output.hidden_states[-sparse_args_dict["feature_layer"]][:, -sparse_args_dict["Txtcls_count"]:]
        global_category_embedding = model.txt_mlp(global_category_embedding)
        global_category_embedding = global_category_embedding.mean(dim=1)

        global_category_embeddings_cache.append(global_category_embedding)

    # 将类别特征向量拼接成 (2, C) 的矩阵，包含气胸和正常两个类别的特征
    global_category_embeddings_cache = torch.cat(global_category_embeddings_cache, dim=0).to(device)

    # 加载问题数据
    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # questions = random.sample(questions, min(100, len(questions)))
    # 存储真实标签和预测结果
    all_labels = []  # 真实标签
    all_probs = []  # 存储类别的预测概率，用于计算 AUC

    # 对每个问题进行推理
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]

        # 创建提示语句
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        input_ids = (
            tokenizer_image_token(
                qs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda(0)
        )
        attention_mask = (input_ids != tokenizer.pad_token_id).long().cuda(0)
        img = pydicom.dcmread(image_file).pixel_array  # 读取 DICOM 图像数据
        img = img.astype(float) / 255.0  # 归一化图像
        img = exposure.equalize_hist(img)  # 直方图均衡化

        # 转换为 PIL 图像并应用预处理
        img = (255 * img).astype(np.uint8)  # 转换为 uint8 类型
        img = Image.fromarray(img).convert('RGB') 
        # image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([img], image_processor, model.config)[0].cuda(0)

        with torch.inference_mode():
            outputs = model.inference_pipeline(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_category_embeddings_cache=global_category_embeddings_cache,
                images=image_tensor.unsqueeze(0).half().cuda(0),
                image_sizes=[img.size],
                use_cache=True,
            )

        # similarity_probs 是一个 (batch_size, 2) 的矩阵，表示气胸和正常两个类别的概率
        similarity_probs = outputs  # 已经 softmax 过了，得到每个类别的预测概率

        # 对于二分类任务，获取气胸类别（pneumothorax）的概率
        probs = similarity_probs[0, 1].item()  # 获取气胸类别的概率（索引1）

        # 获取真实标签，气胸为 1，正常为 0
        true_labels = torch.zeros(1)
        true_labels[0] = 1 if 'pneumonia' in line["question_id"].lower() else 0

        # 将标签和预测概率存储到全局变量
        all_labels.append(true_labels.cpu().numpy())  # 保存真实标签
        all_probs.append(probs)  # 保存预测概率

    # 将 all_labels 和 all_probs 转换为 numpy 数组
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs).squeeze()  # 去除多余的维度，确保是一维数组

    # 计算每个类别的 precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)  # 避免分母为0

    # 选择最大 F1 分数所对应的最佳阈值
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]

    # 使用最佳阈值对预测结果进行二值化
    binary_predictions = (all_probs >= best_threshold).astype(int)

    # 计算准确率（Accuracy）
    accuracy = (binary_predictions == all_labels).mean()

    # 计算 AUC 和 F1 分数
    auc_score = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, binary_predictions)  # 使用最佳阈值二值化后的预测值计算 F1 分数

    # 计算 AUPRC
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    auprc_score = auc(recall, precision)

    # 打印评估指标
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc_score}")
    print(f"AUPRC: {auprc_score}")
    print(f"F1 Score: {f1}")
    print(f"Best Threshold: {best_threshold}")

    # 保存评估结果
    result_metrics = {
        "accuracy": accuracy,
        "auc": auc_score,
        "auprc": auprc_score,
        "f1": f1,  # 添加 F1 分数
        "best_threshold": best_threshold,
    }

    # 检查目录并创建
    result_dir = os.path.dirname(args.result_file)  
    if result_dir and not os.path.exists(result_dir):  
        os.makedirs(result_dir, exist_ok=True)  

    # 写入文件
    with open(args.result_file, 'w') as f:
        for key, value in result_metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Results saved to {args.result_file}")

def eval_model_padchest(args, sparse_args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------- 模型加载与全局类别嵌入计算 ---------------------
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, sparse_args, device_map=None  # 让 DataParallel 处理 device
    )

    # 多 GPU 推理
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])  # 指定 GPU 0 和 1
    model.to(device)
    model.eval()
    if hasattr(model, "module"):
        model_config = model.module.config
    else:
        model_config = model.config


    classes = [
        "Normal", "Pulmonary Fibrosis", "Chronic Changes", "Kyphosis", "Pseudonodule", 
        "Ground Glass Pattern", "Unchanged", "Alveolar Pattern", "Interstitial Pattern", 
        "Laminar Atelectasis", "Pleural Effusion", "Apical Pleural Thickening", "Suture Material", 
        "Sternotomy", "Endotracheal Tube", "Infiltrates", "Heart Insufficiency", "Hemidiaphragm Elevation", 
        "Superior Mediastinal Enlargement", "Aortic Elongation", "Scoliosis", "Sclerotic Bone Lesion", 
        "Supra Aortic Elongation", "Vertebral Degenerative Changes", "Goiter", "COPD Signs", 
        "Air Trapping", "Descendent Aortic Elongation", "Aortic Atheromatosis", "Metal", "Hypoexpansion Basal", 
        "Abnormal Foreign Body", "Central Venous Catheter via Subclavian Vein", "Central Venous Catheter", 
        "Vascular Hilar Enlargement", "Pacemaker", "Atelectasis", "Vertebral Anterior Compression", 
        "Hiatal Hernia", "Pneumonia", "Diaphragmatic Eventration", "Consolidation", "Calcified Densities", 
        "Cardiomegaly", "Fibrotic Band", "Tuberculosis Sequelae", "Volume Loss", "Bronchiectasis", 
        "Single Chamber Device", "Emphysema", "Vertebral Compression", "Bronchovascular Markings", 
        "Bullas", "Hilar Congestion", "Exclude", "Axial Hyperostosis", "Aortic Button Enlargement", 
        "Calcified Granuloma", "Clavicle Fracture", "Pulmonary Mass", "Dual Chamber Device", "Increased Density", 
        "Surgery Neck", "Osteosynthesis Material", "Costochondral Junction Hypertrophy", "Segmental Atelectasis", 
        "Costophrenic Angle Blunting", "Calcified Pleural Thickening", "Hyperinflated Lung", "Callus Rib Fracture", 
        "Pleural Thickening", "Mediastinal Mass", "Nipple Shadow", "Surgery Heart", "Pulmonary Artery Hypertension", 
        "Central Vascular Redistribution", "Tuberculosis", "Nodule", "Cavitation", "Granuloma", "Osteopenia", 
        "Lobar Atelectasis", "Surgery Breast", "NSG Tube", "Hilar Enlargement", "Gynecomastia", "Atypical Pneumonia", 
        "Cervical Rib", "Mediastinal Enlargement", "Major Fissure Thickening", "Surgery", "Azygos Lobe", "Adenopathy", 
        "Miliary Opacities", "Suboptimal Study", "DAI", "Mediastinic Lipomatosis", "Surgery Lung", "Mammary Prosthesis", 
        "Humeral Fracture", "Calcified Adenopathy", "Reservoir Central Venous Catheter", "Vascular Redistribution", 
        "Hypoexpansion", "Heart Valve Calcified", "Pleural Mass", "Loculated Pleural Effusion", "Pectum Carinatum", 
        "Subacromial Space Narrowing", "Central Venous Catheter via Jugular Vein", "Vertebral Fracture", "Osteoporosis", 
        "Bone Metastasis", "Lung Metastasis", "Cyst", "Humeral Prosthesis", "Artificial Heart Valve", "Mastectomy", 
        "Pericardial Effusion", "Lytic Bone Lesion", "Subcutaneous Emphysema", "Edema", "Flattened Diaphragm", 
        "Asbestosis Signs", "Multiple Nodules", "Prosthesis", "Pulmonary Hypertension", "Soft Tissue Mass", 
        "Tracheostomy Tube", "Endoprosthesis", "Post Radiotherapy Changes", "Air Bronchogram", "Pectum Excavatum", 
        "Calcified Mediastinal Adenopathy", "Central Venous Catheter via Umbilical Vein", "Thoracic Cage Deformation", 
        "Obesity", "Tracheal Shift", "External Foreign Body", "Atelectasis Basal", "Aortic Endoprosthesis", 
        "Rib Fracture", "Calcified Fibroadenoma", "Pneumothorax", "Reticulonodular Interstitial Pattern", 
        "Reticular Interstitial Pattern", "Chest Drain Tube", "Minor Fissure Thickening", "Fissure Thickening", 
        "Hydropneumothorax", "Breast Mass", "Blastic Bone Lesion", "Respiratory Distress", "Azygoesophageal Recess Shift", 
        "Ascendent Aortic Elongation", "Lung Vascular Paucity", "Kerley Lines", "Electrical Device", 
        "Artificial Mitral Heart Valve", "Artificial Aortic Heart Valve", "Total Atelectasis", 
        "Non Axial Articular Degenerative Changes", "Pleural Plaques", "Calcified Pleural Plaques", 
        "Lymphangitis Carcinomatosa", "Lepidic Adenocarcinoma", "Mediastinal Shift", "Ventriculoperitoneal Drain Tube", 
        "Esophagic Dilatation", "Dextrocardia", "End On Vessel", "Right Sided Aortic Arch", "Chilaiditi Sign", 
        "Aortic Aneurysm", "Loculated Fissural Effusion", "Fracture", "Air Fluid Level", "Round Atelectasis", 
        "Mass", "Double J Stent", "Pneumoperitoneo", "Abscess", "Pulmonary Artery Enlargement", "Bone Cement", 
        "Pneumomediastinum", "Catheter", "Surgery Humeral", "Empyema", "Nephrostomy Tube", 
        "Sternoclavicular Junction Hypertrophy", "Pulmonary Venous Hypertension", "Gastrostomy Tube", "Lipomatosis"
    ]

 
    categories = [f"This is a chest X-ray showing {category}" for category in classes]

    # 编码类别文本
    encoded_categories = [tokenizer(category, return_tensors="pt") for category in categories]
    category_ids = pad_sequence(
        [item.input_ids.squeeze(0) for item in encoded_categories],
        batch_first=True
    ).to(device)
    category_attention_mask = pad_sequence(
        [item.attention_mask.squeeze(0) for item in encoded_categories],
        batch_first=True
    ).to(device)

    # 计算全局类别特征向量（只计算一次）
    global_category_embeddings_cache = []
    sparse_args_dict = asdict(sparse_args)
    with torch.no_grad():
        for i in range(category_ids.size(0)):
            category_input_ids = category_ids[i].unsqueeze(0).to(device)
            category_attention = category_attention_mask[i].unsqueeze(0).to(device)

            category_output = model(
                input_ids=category_input_ids, 
                attention_mask=category_attention,
                output_hidden_states=True,
                return_emb=True,
                return_dict=True
            )

            # 取最后指定层的隐藏状态，并取末尾 Txtcls_count 个 token
            global_category_embedding = category_output.hidden_states[-sparse_args_dict["feature_layer"]][:, -sparse_args_dict["Txtcls_count"]:]
            global_category_embedding = model.module.txt_mlp(global_category_embedding) if hasattr(model, 'module') else model.txt_mlp(global_category_embedding)
            global_category_embedding = global_category_embedding.mean(dim=1)
            global_category_embeddings_cache.append(global_category_embedding)
    
    global_category_embeddings_cache = torch.cat(global_category_embeddings_cache, dim=0).to(device)

    # --------------------- 数据准备 ---------------------
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # questions = random.sample(questions, min(100, len(questions)))
    all_labels = []
    all_probs = []

    # --------------------- 推理与指标计算 ---------------------
    for line in tqdm(questions):
        image_file = args.image_folder + line["image"]
        qs = line["text"]

        if model_config.mm_use_im_start_end:
            qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs)
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

        img_path = os.path.join(args.image_folder, image_file)
        
        # 尝试加载图像，如果遇到异常则跳过该图像
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model_config)[0].to(device)
        except Exception as e:
            print(f"Warning: Skipping image {img_path} due to error: {e}")
            continue  # 跳过当前图像，继续下一个图像

        # **关键修改：使用多 GPU 进行推理**
        with torch.inference_mode():
            outputs = model.module.inference_pipeline(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                global_category_embeddings_cache=global_category_embeddings_cache,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                use_cache=True,
            ) if hasattr(model, 'module') else model.inference_pipeline(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                global_category_embeddings_cache=global_category_embeddings_cache,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                use_cache=True,
            )

        similarity_probs = outputs

        # **关键修改：跳过异常图像时，同步跳过对应标签**
        true_labels = torch.zeros(len(classes))
        label_dict = line["label"]
        for disease, value in label_dict.items():
            if value == 1 and disease in classes:
                true_labels[classes.index(disease)] = 1

        # 如果图像没有被跳过，则记录预测和标签
        all_labels.append(true_labels.cpu().numpy())
        all_probs.append(similarity_probs.cpu().numpy())

    # 计算性能指标时，确保只有有效数据
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs).squeeze(1)

    # --------------------- 计算性能指标 ---------------------
    accuracies, auc_scores, auprc_scores, f1_scores, precision_scores, recall_scores = [], [], [], [], [], []

    for i in range(all_labels.shape[1]):
        precision_vals, recall_vals, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        f1 = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-8)
        max_f1_idx = np.argmax(f1)
        best_threshold = thresholds[max_f1_idx]
        predictions_binary = (all_probs[:, i] >= best_threshold).astype(int)
        accuracies.append((predictions_binary == all_labels[:, i]).mean())

        try:
            auc_scores.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
        except ValueError:
            auc_scores.append(np.nan)

        auprc_scores.append(auc(recall_vals, precision_vals))
        f1_scores.append(np.max(f1))
        precision_scores.append(precision_vals[max_f1_idx])
        recall_scores.append(recall_vals[max_f1_idx])

    result_metrics = {
        "mean_accuracy": np.mean(accuracies),
        "mean_auc": np.nanmean(auc_scores),
        "mean_f1": np.mean(f1_scores),
        "mean_auprc": np.mean(auprc_scores),
        "mean_precision": np.mean(precision_scores),
        "mean_recall": np.mean(recall_scores),
    }

    print("\n===== Evaluation Metrics =====")
    for key, value in result_metrics.items():
        print(f"{key}: {value}")

    result_dir = os.path.dirname(args.result_file)
    os.makedirs(result_dir, exist_ok=True) if result_dir and not os.path.exists(result_dir) else None

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
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/srv/lby/llava_med/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--result-folder", type=str, default="./result/R4090/llava-mistral_new_clip_v9/")
    parser.add_argument("--dataset", type=str, default="siim")
    parser.add_argument("--subdata", type=str, default="unseen")
    parser.add_argument("--chexpert-subset", type=str)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args, remaining_args = parser.parse_known_args()
    
    # Use HfArgumentParser for SparseArguments
    hf_parser = HfArgumentParser(SparseArguments)
    sparse_args, = hf_parser.parse_args_into_dataclasses(remaining_args)

    test(args, sparse_args)
    # eval_model_chest_xray(args, sparse_args)
    # eval_model_SIIM(args, sparse_args)
    # eval_model_chexpert(args, sparse_args)
    # eval_model_rsna(args, sparse_args)
    # eval_model_padchest(args, sparse_args)