import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.grad
from transformers import AutoTokenizer
from transformers import AutoConfig, AutoModelForCausalLM
# from transformers import AutoConfig, AutoModelForCausalLM, \
#                          MistralConfig, MistralModel, MistralForCausalLM
from llava.model.modeling_mistral import (
    MistralConfig,
    MistralModel,
    MistralForCausalLM
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
# from llava.constants import NUM_LABELS
from deepspeed.utils import init_on_device
from deepspeed import zero

from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch.distributed import get_world_size, get_rank, group
import torch.distributed as dist




class LlavaMistralConfig(MistralConfig):
    model_type = "llava_mistral"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 确保调用父类的初始化
       


    
class LlavaMistralModel(LlavaMetaModel, MistralModel):
    config_class = LlavaMistralConfig

    def __init__(self, config: MistralConfig):
        super(LlavaMistralModel, self).__init__(config)
        

      
    
class linea_layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(linea_layer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)  
          
class mis_mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mis_mlp, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x 


class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMistralConfig

    def __init__(self, config, ncls_token_id):
        
        super(MistralForCausalLM, self).__init__(config)
        self.model = LlavaMistralModel(config)
        # self.tokenizer = tokenizer
        self.padding_idx = config.pad_token_id

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mis_mlp = None
        self.ncls_token_id = ncls_token_id
        self.ncls_count = 8  # 可以根据需要调整

        # Initialize weights and apply final processing
        self.post_init()
    
    def initialize_mis_mlp(self):
        """确保 mis_mlp 的初始化在模型权重加载后完成"""
        if self.mis_mlp is None:
            self.mis_mlp = mis_mlp(input_dim=4096, hidden_dim=1024, output_dim=512)
            # 注册到模块列表中
            self.add_module("mis_mlp", self.mis_mlp)
            # for param in self.mis_mlp.parameters():
            #     param.requires_grad = True
                
    def update_tensor(self, batch_size, tensor, ncls_tensor):
    # 直接在序列末尾追加 ncls 标记
        updated_tensors = []
        for i in range(batch_size):
            updated_tensor = torch.cat([tensor[i], ncls_tensor[i]], dim=0)
            updated_tensors.append(updated_tensor)
        return torch.stack(updated_tensors)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, 
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        category_ids: Optional[torch.LongTensor] = None,  # 新增类别文本的输入
        category_attention_mask: Optional[torch.Tensor] = None,  # 新增类别文本的注意力mask
        txt_input_ids: torch.LongTensor = None, # 新增文本输入
        txt_attention_mask: Optional[torch.Tensor] = None,  # 新增文本的注意力mask
        txt_position_ids: Optional[torch.LongTensor] = None, # 新增文本的位置编码
        txt_past_key_values: Optional[List[torch.FloatTensor]] = None, # 新增文本的过去的 key-values
        txt_inputs_embeds: Optional[torch.FloatTensor] = None,  # 新增文本的嵌入
        return_emb: Optional[bool] = False,  # 新增是否返回嵌入的标志(不计算loss)
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if input_ids is not None and attention_mask is not None:
            batch_size = input_ids.size(0)

            # 创建 ncls 标记
            ncls_token_ids = torch.full((batch_size, self.ncls_count), self.ncls_token_id, dtype=input_ids.dtype, device=input_ids.device)
            ncls_attention_mask = torch.ones((batch_size, self.ncls_count), dtype=attention_mask.dtype, device=attention_mask.device)

            # 直接在序列末尾添加 ncls 标记
            input_ids = self.update_tensor(batch_size, input_ids, ncls_token_ids)
            attention_mask = self.update_tensor(batch_size, attention_mask, ncls_attention_mask)

            # 更新 position_ids
            if position_ids is None:
                position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            else:
                # 添加位置标记到末尾
                ncls_position_ids = torch.arange(position_ids.size(1), position_ids.size(1) + self.ncls_count, device=position_ids.device).unsqueeze(0).expand(batch_size, -1)
                position_ids = self.update_tensor(batch_size, position_ids, ncls_position_ids)

            # 更新 labels
            if labels is not None:
                ncls_labels = torch.full((batch_size, self.ncls_count), -100, dtype=labels.dtype, device=labels.device)  # -100 用于忽略 ncls 部分
                labels = self.update_tensor(batch_size, labels, ncls_labels)

            ###更新文本部分 填充ncls标记
            # txt_input_ids = input_ids.clone() 
            # txt_attention_mask = attention_mask.clone()
            # txt_past_key_values = past_key_values
            # txt_labels = labels
        
        # 更新类别输入
        if category_ids is not None and category_attention_mask is not None:
            category_valid_lengths = category_attention_mask.sum(dim=1).long()
            category_ncls_token_ids = torch.full((batch_size, self.ncls_count), self.ncls_token_id, dtype=category_ids.dtype, device=category_ids.device)
            category_ncls_attention_mask = torch.ones((batch_size, self.ncls_count), dtype=category_attention_mask.dtype, device=category_attention_mask.device)
            
            # 直接在类别输入的序列末尾添加 ncls 标记
            category_ids = self.update_tensor(batch_size, category_ids.squeeze(1), category_ncls_token_ids)
            category_attention_mask = self.update_tensor(batch_size, category_attention_mask.squeeze(1), category_ncls_attention_mask)
        
        # 推理时处理所有类别矩阵
        if return_emb:
            txt_input_ids = input_ids.clone() 
            txt_attention_mask = attention_mask.clone()
            txt_past_key_values = past_key_values
        else:
            txt_input_ids = category_ids
            txt_attention_mask = category_attention_mask
            txt_past_key_values = past_key_values
      
            
        # 如果仅有图像输入，则使用图像的多模态预处理
        if inputs_embeds is None and images is not None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes
            )
        
        # 如果仅有类别输入，则处理类别文本的嵌入
        if txt_input_ids is not None:
            txt_input_ids, txt_position_ids, txt_attention_mask, txt_past_key_values, txt_inputs_embeds = self.prepare_inputs_txt_labels_for_multimodal(
                txt_input_ids, txt_position_ids, txt_attention_mask, txt_past_key_values
                )
        
        
        # 获取模型的文本输出
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            txt_input_ids=txt_input_ids,
            txt_position_ids=txt_position_ids,
            txt_attention_mask=txt_attention_mask,
            txt_past_key_values=txt_past_key_values,
            txt_inputs_embeds=txt_inputs_embeds,
            return_emb=return_emb
        )
        

        return outputs

    def get_model(self):
        # 假设你返回一个基础模型或者相关的模型实例
        return self.base_model
    
    
    @torch.no_grad()
    def inference_pipeline(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        category_embeddings_cache: torch.Tensor,  # 新增参数，用于传递类别特征向量
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Step 2: 获取图片特征向量
        image_output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_sizes=image_sizes,
            output_hidden_states=True,
            return_emb = True,
            return_dict=True
        )
        # print('------------')
        # print(image_output)
        image_embedding = image_output.hidden_states[-1][:, -self.ncls_count:]
        image_embedding = torch.max(image_embedding, dim=1).values  # 使用最大池化

        # 步骤2: 对图像特征和类别特征进行L2归一化
        image_embedding = F.normalize(image_embedding, p=2, dim=-1)
        category_embeddings_cache = F.normalize(category_embeddings_cache, p=2, dim=-1)

        # 步骤3: 计算余弦相似度矩阵
        similarity_matrix = torch.matmul(image_embedding, category_embeddings_cache.T)  # 计算余弦相似度

        # 步骤4: 将相似度矩阵转换为概率分布 (如果需要的话)
        similarity_probs = similarity_matrix.softmax(dim=-1)

        # 步骤5: 获取相似度最高的 top-k 类别
        # topk_values, topk_indices = similarity_probs.topk(5, dim=-1)

        # 返回结果
        return similarity_probs
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs


AutoConfig.register("llava_mistral", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)
