'''
Author: fly
Date: 2024-08-23 15:47:41
FilePath: /llava_probe/home/lby/llava_med/LLaVA-Med/llava/run/train/train_mem.py
Description: 
'''
from llava.run.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
