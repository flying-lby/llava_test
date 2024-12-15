import os
import numpy as np
import torch
import json
import math
import random
import click
from tqdm import trange
from matplotlib import pyplot as plt


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


@click.command()
@click.option("--dataset", default="Chest-X-ray")
@click.option("--model_name", default="llava_med")
@click.option("--probe", default="linear")
@click.option("--split", default="test")
@click.option("--class_path", default="/home/lby/llava_med/LLaVA-Med/llava/run/data/eval/Chest-X-ray_classes.json")
@click.option("--feature_type", default="last")
@click.option("--n_epochs", default=500)
def main(dataset,class_path,model_name, probe, split, feature_type, n_epochs):
    data = [json.loads(line) for line in open(f"/home/lby/llava_med/LLaVA-Med/llava/run/data/eval/{dataset}.jsonl")]
    feature = torch.load(f"/home/lby/llava_med/outputs_{i}.pt")
    random.seed(1234)
    random.shuffle(data)
    classes =  json.load(open(class_path))

    labels = []
    for item in data:
        # 将字符串形式的标签转换为列表，去掉空格
        item_labels = [label.strip() for label in item["label"].split(',')]
        
        # 创建一个与 classes 等长的零向量
        label_indices = [0] * len(classes)
        
        # 遍历分割后的标签列表，并标记对应的索引为 1
        for label in item_labels:
            if label in classes:
                label_indices[classes.index(label)] = 1
            else:
                raise ValueError(f"标签 {label} 不在 classes 列表中")
        
        # 保存该样本的标签索引
        item["label_index"] = label_indices
        labels.append(label_indices)

    # 将 labels 列表转换为 PyTorch 张量
    labels = torch.tensor(labels)

    n_shards = math.ceil(len(data) // 1024) + 2
    print(len(data), n_shards)

    features = []
    for i in range(n_shards):
        # feature = torch.load(f"outputs/{dataset}_{model_name}_{i}.pt")
        
        features.append(feature)

    features = torch.cat(features, dim=0).float()
    print(features.shape)
    
    train_idxs = [i for i in range(len(data)) if data[i]["split"] == "train"]
    test_idxs = [i for i in range(len(data)) if data[i]["split"] == split]

    feature_idx = None
    if feature_type == "last":
        feature_idx = 0
    elif feature_type == "avg":
        feature_idx = 1
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    print(len(data))  # 确保 data 长度正确
    print(features.shape[0])  # 确保 features 长度一致
    train_features = features[train_idxs, feature_idx]
    test_features = features[test_idxs, feature_idx]
    train_labels = labels[train_idxs].float()  # 标签为 float
    test_labels = labels[test_idxs].float()

    print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

    if probe == "linear":
        model = LinearProbing(len(train_features[0]), len(classes)).cuda()
    elif probe == "mlp":
        model = MLPProbing(len(train_features[0]), len(classes)).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()  # 使用二元交叉熵损失

    bsz = 512

    accs = []
    for epoch in trange(n_epochs):
        model.train()  # 设置模型为训练模式
        for i in range(0, len(train_features), bsz):
            optimizer.zero_grad()
            output = model(train_features[i : i + bsz].cuda())
            loss = criterion(output, train_labels[i : i + bsz].cuda())
            loss.backward()
            optimizer.step()

        model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            eval_bsz = 512

            # 训练集评估
            preds = []
            for i in range(0, len(train_features), eval_bsz):
                output = model(train_features[i : i + eval_bsz].cuda())
                pred = torch.sigmoid(output).cpu()  # 将 logits 转化为概率
                preds.append(pred)
            preds = torch.cat(preds)
            train_acc = ((preds > 0.5) == train_labels).float().mean().item()  # 使用 0.5 阈值计算准确率

            # 测试集评估
            preds = []
            for i in range(0, len(test_features), eval_bsz):
                output = model(test_features[i : i + eval_bsz].cuda())
                pred = torch.sigmoid(output).cpu()  # 将 logits 转化为概率
                preds.append(pred)
            preds = torch.cat(preds)
            test_acc = ((preds > 0.5) == test_labels).float().mean().item()  # 使用 0.5 阈值计算准确率

            accs.append((train_acc, test_acc))

    plt.plot([train_acc for train_acc, _ in accs], label="train")
    plt.plot([test_acc for _, test_acc in accs], label="test")
    plt.legend()

    output_prefix = f"probe_outputs/{dataset}_{model_name}_{probe}_{split}_{feature_type}"
    output_dir = os.path.dirname(output_prefix)

    # 如果目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存图像
    plt.savefig(f"{output_prefix}.png")
    for epoch, (train_acc, test_acc) in enumerate(accs):
        print(f"Epoch {epoch + 1}:")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Testing Accuracy: {test_acc:.4f}")

    # 保存模型和准确率
    torch.save([accs, model.state_dict()], f"{output_prefix}.pt")



if __name__ == "__main__":
    main()


   