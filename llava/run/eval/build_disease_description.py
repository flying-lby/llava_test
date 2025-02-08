import argparse
import json
import os
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from tqdm import tqdm
import torch
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
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from dataclasses import dataclass
from dataclasses import asdict
from transformers import HfArgumentParser
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, recall_score, f1_score, roc_auc_score
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import re
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


@dataclass
class SparseArguments:
    Imgcls_count: int = 4
    Txtcls_count: int = 8
    hidden_dim: int = 1024
    output_dim: int = 512
    img_mlp_type: int = 0
    txt_mlp_type: int = 0
    loss_threshold: float = 0.5
    temperature: float = 0.05
    use_local_loss: bool = False
    feature_layer: int = 1
    special_tokens_mlp_type: int = 1
    use_ca_loss: bool = True
    inference_type: int = 2
    use_cat: bool = True
    use_prompt: bool = True

def build_model(args):
    disable_torch_init()

    # List of diseases
    disease_list = [
        'fibrosis', 'edema', 'pneumothorax', 'cardiomegaly', 'atelectasis', 'nodule',
        'emphysema', 'no finding', 'mass', 'pleural_thickening', 'effusion', 
        'infiltration', 'pneumonia', 'hernia', 'consolidation'
    ]

    # Prompt template with Chain-of-Thought (CoT) strategy
    PROMPT_TEMPLATE = """You are a senior medical imaging expert. Please provide a concise description of the disease "{disease}" in the following format:
                    - **Disease Description**: Provide a brief, clear explanation of the disease, including its pathology and clinical significance.
                    - **Imaging Characteristics**: Describe the typical imaging features, such as lesion location, shape, boundaries, and density variations.
                    - **Clinical Presentation**: Mention common clinical symptoms or signs that help diagnose the disease.
                    Given an Example: Emphysema is a chronic obstructive pulmonary disease caused by the permanent destruction of alveolar walls and airspace enlargement. Imaging shows scattered or diffuse low-density areas in both lungs, reduced lung markings, often with bullae or cystic lesions, flattened diaphragm, and hyperinflated lungs. Clinically, patients typically have a history of chronic cough, sputum production, and progressive dyspnea, often associated with smoking or long-term occupational exposure.
                """

    # Initialize LLaVA model
    model_path = '/srv/lby/llava_med/llava-med-v1.5-mistral-7b'
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map='cuda:0'
    )
    
    # Ensure pad_token_id is correctly set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


    disease_descriptions = {}

    for disease in disease_list:
        # Format prompt with disease name
        prompt = f"[INST]{PROMPT_TEMPLATE.format(disease=disease)}[/INST]"

        # Tokenize input prompt
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # Generate response using model
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_length=input_ids.shape[1] + args.max_new_tokens,  # Ensure this is large enough for the full output
                use_cache=True,
            )

        # Decode the output tokens and clean up formatting issues
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
       
        disease_descriptions[disease] = outputs
       

    # Save the dictionary to a JSON file
    import json
    with open('disease_descriptions.json', 'w') as f:
        json.dump(disease_descriptions, f, ensure_ascii=False, indent=4)
    
    
    
def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map='cuda:0'
    )

    qs = args.query
   

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    print(prompt)
  

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=None,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    args = parser.parse_args()

    # eval_model(args)
    # parser = argparse.ArgumentParser()
    # args, remaining_args = parser.parse_known_args()
    # # Use HfArgumentParser for SparseArguments
    # hf_parser = HfArgumentParser(SparseArguments)
    # sparse_args, = hf_parser.parse_args_into_dataclasses(remaining_args)

    build_model(args)
    