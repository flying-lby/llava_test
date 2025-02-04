'''
Author: fly
Date: 2024-08-08 13:56:14
FilePath: /llava_med/LLaVA-Med/llava/model/builder.py
Description: 
'''
import os
import shutil
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import LlavaMistralForCausalLM
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from dataclasses import dataclass

# 定义 SparseArguments 数据类
@dataclass
class SparseArguments:
    Imgcls_count: int = 4
    Txtcls_count: int = 4
    hidden_dim: int = 1024
    output_dim: int = 512
    img_mlp_type: int = 1
    txt_mlp_type: int = 1
    loss_threshold: float = 0.5
    temperature: float = 0.05
    use_local_loss: bool = False
    feature_layer: int = 1
    special_tokens_mlp_type: int = 1
    use_ca_loss: bool = True
    inference_type: int = 2
    use_cat: bool = True
    use_prompt: bool = True
   

# 全局定义 add_sparse 参数
default_sparse_args = SparseArguments()

# torch.cuda.set_device(1)
def load_pretrained_model(model_path, model_base, model_name, add_sparse=None, load_8bit=False, load_4bit=False, device_map="auto", device="cuda:0", use_flash_attn=False):
    
    # 如果没有传递 add_sparse，则使用全局定义的默认值
    if not add_sparse:
        add_sparse = default_sparse_args

        
    kwargs = {}
    
    # 如果设备不是 CUDA，设置设备映射
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    else:
        kwargs['device_map'] = device_map

    # 设置低内存使用选项
    # kwargs['low_cpu_mem_usage'] = True

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            
            from llava.model.language_model.llava_mistral import LlavaMistralConfig
            lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            
            #-----------------------调参-------------------------------------------------#
        
            lora_cfg_pretrained.sparse_config = vars(add_sparse)

            #------------------------------------------------------------------------#
            
            # 定义特殊标记
            Imgcls_token = "<Imgcls>"
            Txtcls_token = "<Txtcls>"

            # 确保特殊标记已正确添加到分词器
            special_tokens = [Imgcls_token, Txtcls_token]

            for token in special_tokens:
                if token not in tokenizer.get_vocab():
                    print(f"Adding '{token}' to tokenizer...")
                    tokenizer.add_tokens([token])
                else:
                    print(f"'{token}' already exists in tokenizer.")

            # 获取并输出特殊标记的 ID
            Imgcls_token_id = tokenizer.convert_tokens_to_ids(Imgcls_token)
            Txtcls_token_id = tokenizer.convert_tokens_to_ids(Txtcls_token)

            print(f"Imgcls_token_id: {Imgcls_token_id}")
            print(f"Txtcls_token_id: {Txtcls_token_id}")

            # 检查分词器词汇表大小
            print(f"Tokenizer vocab size after adding special tokens: {len(tokenizer)}")

            # 加载 Llava 模型
            print("Loading LLaVA model with customized configurations...")
            model = LlavaMistralForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=lora_cfg_pretrained,
                Imgcls_token_id=Imgcls_token_id,
                Txtcls_token_id=Txtcls_token_id,
                **kwargs
            )
            
            # 调整词汇表大小，并确保只新增标记的权重被初始化
            model.resize_token_embeddings(len(tokenizer))
           
            # 加载训练后的权重
            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

            # 加载权重
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)
            
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
                        
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                # model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                # model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                Imgcls_token = "<Imgcls>"
                Txtcls_token = "<Txtcls>"
                
                special_tokens = [Imgcls_token, Txtcls_token]
                for token in special_tokens:
                    if token not in tokenizer.get_vocab():
                        print(f"Adding '{token}' to tokenizer...")
                        tokenizer.add_tokens([token])
                    else:
                        print(f"'{token}' already exists in tokenizer.")

                # 获取并输出  ID
                Imgcls_token_id = tokenizer.convert_tokens_to_ids(Imgcls_token)
                Txtcls_token_id = tokenizer.convert_tokens_to_ids(Txtcls_token)
       
                #-----------------------调参-------------------------------------------------#
             
              
                from llava.model.language_model.llava_mistral import LlavaMistralConfig
                config = LlavaMistralConfig.from_pretrained(model_path)
                if add_sparse:
                    config.sparse_config = vars(add_sparse)
               
                #------------------------------------------------------------------------#
                
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    Imgcls_token_id=Imgcls_token_id,
                    Txtcls_token_id=Txtcls_token_id,
                    config =config,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
           
              
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                # model = LlavaLlamaForCausalLM.from_pretrained(
                #     model_path,
                #     low_cpu_mem_usage=True,
                #     **kwargs
                # )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
            # vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len





