import os
import json
import random
import shutil

# 路径设置
input_json_path = './data/chest_xray/new_classify_mimic_file_clip.json'
output_json_path = 'test.json'
image_root_dir = '/mnt/nlp-ali/usr/huangwenxuan/home/dataset/srv/lby/physionet.org/files/mimic-cxr-jpg/2.0.0/files'  # 原始图片的根目录
output_image_dir = 'test'  # 要保存图片的目标根目录

# 创建输出根目录
os.makedirs(output_image_dir, exist_ok=True)

# 读取 JSON 数据
with open(input_json_path, 'r') as f:
    data = json.load(f)

# 随机选取 100 条数据
selected_data = random.sample(data, 100)

# 复制图片并保持目录结构
for item in selected_data:
    rel_path = item['image']  # 保持不变
    src_path = os.path.join(image_root_dir, rel_path)
    dst_path = os.path.join(output_image_dir, rel_path)

    # 创建必要的子目录
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # 拷贝图片
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"警告：未找到图片 {src_path}")

# 保存选中的 JSON 数据
with open(output_json_path, 'w') as f:
    json.dump(selected_data, f, indent=4)
