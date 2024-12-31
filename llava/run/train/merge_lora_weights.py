'''
Author: fly
Date: 2024-09-01 21:16:43
FilePath: /llava_med/LLaVA-Med/llava/run/train/merge_lora_weights.py
Description: 
'''
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch
from dataclasses import dataclass
import argparse
from transformers import HfArgumentParser

@dataclass
class SparseArguments:
    ncls_count: int = 4
    hidden_dim: int = 1024
    output_dim: int = 512
    mlp_type: int = 0
    loss_threshold: float = 0.5
    temperature: float = 0.05
    use_local_loss: bool = True
    feature_layer: int = 1
    

def merge_lora(args, sparse_args):
    # 设置 GPU 0 为当前设备
    torch.cuda.set_device(0)
    
    # 获取模型名称
    model_name = get_model_name_from_path(args.model_path)

    # 加载模型到 GPU 0
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, sparse_args,  device_map='cuda:0')

    # 保存模型和分词器
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, default= "/home/lby/llava_med/LLaVA-Med/llava/run/checkpoints/llava-lora-new-clip-version4")
    parser.add_argument("--model-base", type=str, required=True, default="/srv/lby/llava_med/llava-med-v1.5-mistral-7b")
    parser.add_argument("--save-model-path", type=str, required=True, default= "/srv/lby/llava_med/checkpoints/llava-mistral_new_clip_ft4")

    args, remaining_args = parser.parse_known_args()
    
    # Use HfArgumentParser for SparseArguments
    hf_parser = HfArgumentParser(SparseArguments)
    sparse_args, = hf_parser.parse_args_into_dataclasses(remaining_args)

    merge_lora(args, sparse_args)
