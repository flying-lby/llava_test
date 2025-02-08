'''
Author: fly
Date: 2024-09-09 13:35:10
FilePath: /llava_med/LLaVA-Med/llava/data_analysis/clip_extract_feature.py
Description: 
'''
import numpy as np

from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, accuracy_score
import torch
from PIL import Image
import argparse
import json
from tqdm import tqdm
import random
import click
import os
from llava.model.clip_llava_builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
import math
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.clip_llava_builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from matplotlib import pyplot as plt
from tqdm import trange

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class LinearProbing(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearProbing, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPProbing(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MLPProbing, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim, feature_dim)
        self.fc2 = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def get_text_feature(model_id,num_chunks,chunk_idx,question_file,answers_file,image_folder,conv_mode):
   
    # processor = AutoProcessor.from_pretrained(model_id)
    # model = LlavaForConditionalGeneration.from_pretrained(
    #     model_id, device_map="auto", torch_dtype=torch.bfloat16
    # )

    all_features = [] 
    tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_id,
            model_base=None,
            model_name='llava-med-v1.5-mistral-7b'
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    questions = get_chunk(questions, num_chunks, chunk_idx)

    # 过滤掉非 "train" split 的样本
    # train_questions = [q for q in questions if q.get("split") == "train"]

    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):  # 遍历过滤后的 train 数据
        idx = line["question_id"]
        image_file = os.path.join(image_folder, line["image"])
        
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

        image = Image.open(os.path.join(image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]
        if len(image_tensor.shape) == 3:  # if it is [3, height, width]
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
        inputs = {
            "input_ids": input_ids,
            "images": image_tensor.half()  # Handled as float16
        }
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            last_features = hidden_states[:, -1, :].unsqueeze(1)
            mean_features = hidden_states.mean(dim=1).unsqueeze(1)
            features = torch.cat([last_features, mean_features], dim=1)
        all_features.append(features.cpu())
     
    all_features = torch.cat(all_features, dim=0)
    torch.save(all_features, "/srv/lby/llava_med/feature_data/features.pt")
    print(f"Features saved to features.pt")
    
    return all_features
    
def compute_AUCs(gt, pred, n_class):
    """计算每个标签的 AUC"""
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def main(dataset, model_name, probe, split, feature_type, n_epochs):
    data = [json.loads(line) for line in open(f"data/eval/{dataset}.jsonl")]
    random.seed(1234)
    random.shuffle(data)
    classes = json.load(open(f"data/eval/{dataset}_classes.json"))

    labels = []
    for item in data:
        item_labels = [label.strip() for label in item["label"].split(',')]
        label_indices = [0] * len(classes)
        for label in item_labels:
            if label in classes:
                label_indices[classes.index(label)] = 1
            else:
                raise ValueError(f"标签 {label} 不在 classes 列表中")
        item["label_index"] = label_indices
        labels.append(label_indices)

    labels = torch.tensor(labels).float()
    all_feature = torch.load("/srv/lby/llava_med/feature_data/features.pt")
    features = all_feature.float()

    train_idxs = [i for i in range(len(data)) if data[i]["split"] == "train"]
    test_idxs = [i for i in range(len(data)) if data[i]["split"] == split]

    feature_idx = 0 if feature_type == "last" else 1
    train_features = features[train_idxs, feature_idx]
    test_features = features[test_idxs, feature_idx]
    train_labels = labels[train_idxs]
    test_labels = labels[test_idxs]

    if probe == "linear":
        model = LinearProbing(len(train_features[0]), len(classes)).cuda()
    elif probe == "mlp":
        model = MLPProbing(len(train_features[0]), len(classes)).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    bsz = 512
    accs = []

    for epoch in trange(n_epochs):
        model.train()
        for i in range(0, len(train_features), bsz):
            optimizer.zero_grad()
            output = model(train_features[i : i + bsz].cuda())
            loss = criterion(output, train_labels[i : i + bsz].cuda())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            eval_bsz = 512
            train_preds, test_preds = [], []

            for i in range(0, len(train_features), eval_bsz):
                output = model(train_features[i : i + eval_bsz].cuda())
                pred = torch.sigmoid(output).cpu()
                train_preds.append(pred)
            train_preds = torch.cat(train_preds)

            for i in range(0, len(test_features), eval_bsz):
                output = model(test_features[i : i + eval_bsz].cuda())
                pred = torch.sigmoid(output).cpu()
                test_preds.append(pred)
            test_preds = torch.cat(test_preds)

            # 计算多标签 AUC
            train_AUROCs = compute_AUCs(train_labels, train_preds, len(classes))
            test_AUROCs = compute_AUCs(test_labels, test_preds, len(classes))
            train_AUROC_avg = np.mean(train_AUROCs)
            test_AUROC_avg = np.mean(test_AUROCs)

            # 计算多标签 F1 和 ACC
            train_f1s, train_accs = [], []
            test_f1s, test_accs = [], []

            for i in range(len(classes)):
                # 训练集的 F1 和 ACC
                precision, recall, thresholds = precision_recall_curve(train_labels[:, i].cpu(), train_preds[:, i].cpu())
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                max_f1_thresh = thresholds[np.argmax(f1_scores)]
                train_f1s.append(np.max(f1_scores))
                train_accs.append(accuracy_score(train_labels[:, i].cpu(), train_preds[:, i].cpu() > max_f1_thresh))

                # 输出训练集每个类别的正类预测数量
                # train_positive_predictions = (train_preds[:, i].cpu()).sum().item()
                # print(f"Class {classes[i]} - Train Positive Predictions: {train_positive_predictions}")

                # 测试集的 F1 和 ACC
                precision, recall, thresholds = precision_recall_curve(test_labels[:, i].cpu(), test_preds[:, i].cpu())
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                max_f1_thresh = thresholds[np.argmax(f1_scores)]
                test_f1s.append(np.max(f1_scores))
                test_accs.append(accuracy_score(test_labels[:, i].cpu(), test_preds[:, i].cpu() > max_f1_thresh))

                # 输出测试集每个类别的正类预测数量
                # test_positive_predictions = (test_preds[:, i].cpu()).sum().item()
                # print(f"Class {classes[i]} - Test Positive Predictions: {test_positive_predictions}")
            train_f1_avg = np.mean(train_f1s)
            test_f1_avg = np.mean(test_f1s)
            train_acc_avg = np.mean(train_accs)
            test_acc_avg = np.mean(test_accs)

            accs.append((train_acc_avg, test_acc_avg))

            # 打印 AUC 和准确率
            print(f"Epoch {epoch + 1}: Train AUROC: {train_AUROC_avg:.4f}, Test AUROC: {test_AUROC_avg:.4f}")
            print(f"Epoch {epoch + 1}: Train ACC: {train_acc_avg:.4f}, Test ACC: {test_acc_avg:.4f}")
            # 打印平均 F1 分数
            train_f1_avg = np.mean(train_f1s)
            test_f1_avg = np.mean(test_f1s)
            print(f"Epoch {epoch + 1}: Train F1 Avg: {train_f1_avg:.4f}, Test F1 Avg: {test_f1_avg:.4f}")

    # 绘制每个 epoch 的平均准确率变化图
    mean_train_acc = [train_acc for train_acc, _ in accs]
    mean_test_acc = [test_acc for _, test_acc in accs]

    plt.plot(mean_train_acc, label="Mean Train Accuracy")
    plt.plot(mean_test_acc, label="Mean Test Accuracy")
    plt.legend()

    output_prefix = f"probe_outputs/{dataset}_{model_name}_{probe}_{split}_{feature_type}"
    plt.savefig(f"{output_prefix}.png")
    torch.save([accs, model.state_dict()], f"{output_prefix}.pt")
    
    


@click.command()
@click.option("--model_id", default="/srv/lby/llava_med/checkpoints/llava-mistral_clip_ft1")
@click.option("--num-chunks", default=1)
@click.option("--chunk-idx", default=0)
@click.option("--image-folder", default="/srv/lby")
@click.option("--conv-mode", default="vicuna_v1")
@click.option("--question-file", default="./data/eval/test_prompt/Chest-X-ray_llava_val.jsonl")
@click.option("--answers-file", default="./data/eval/test_prompt/Chest-X-ray_llava_val_ans.jsonl")
# @click.option("--class_path", default="./data/eval/Chest-X-ray_classes.json")
# @click.option("--seed", default=1234)
# @click.option("--output_path", default="outputs")
# @click.option("--batch_size", default=2)
@click.option("--dataset", default="Chest-X-ray")
@click.option("--model_name", default="llava-mistral_ft2")
@click.option("--probe", default="linear")
@click.option("--split", default="test")
@click.option("--feature_type", default="last")
@click.option("--n_epochs", default=500)
# @click.option("--prompt", default=None)
def entry(model_id, num_chunks, chunk_idx, image_folder, conv_mode, question_file, answers_file, dataset, model_name, probe, split, feature_type, n_epochs):
# def entry(dataset, model_name, probe, split, feature_type, n_epochs):

    # all_feature = get_feature(model_id, num_chunks, chunk_idx, question_file, answers_file, image_folder, conv_mode)
    main(dataset, model_name, probe, split, feature_type, n_epochs)
 
if __name__ == "__main__":
    entry()