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
          
# class mis_mlp(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(mis_mlp, self).__init__()
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)
#         return x 

class mis_mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mlp_type):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mlp_type = mlp_type
        
        if(self.mlp_type == 1):
            self.out_mlp = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        elif(self.mlp_type == 2):
            self.out_mlp = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim , self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        elif(self.mlp_type == 3):
            self.out_mlp = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.input_dim // 2),
                nn.GELU(),
                nn.Linear(self.input_dim // 2, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        elif(self.mlp_type == 4):
            self.out_mlp = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Dropout(0.3),
                nn.Linear(self.input_dim, self.output_dim),
            )
        elif(self.mlp_type == 5):
            self.out_mlp = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Dropout(0.7),
                nn.Linear(self.input_dim, self.output_dim),
            )
        elif(self.mlp_type == 6):
            self.out_mlp = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.output_dim),
            )
        else:
            # When mlp_type is 0, set self.out_mlp to None
            self.out_mlp = None
      

    def forward(self, x):
        if self.out_mlp is None:
            # If mlp_type is 0, return x directly
            return x
        return self.out_mlp(x) 
    
    

class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_size, num_heads=32, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout)
    
    def forward(self, query_features, key_value_features):
        # query_features: (B, 4096)
        # key_value_features: (B, seq_len, 4096)
        
        query = query_features.unsqueeze(0)  # (1, B, hidden_size)
        key_value = key_value_features.unsqueeze(0)  # (1, B, hidden_size)
        
        # MultiheadAttention 输入必须为 (seq_len, batch_size, embed_dim)
        attn_output, attn_weights = self.attention(query, key_value, key_value)  # (1, B, hidden_size), (B, 1, 1)
        
        # 返回输出到 (B, hidden_size)
        attn_output = attn_output.squeeze(0)  # (B, hidden_size)
        
        return attn_output, attn_weights
    
    

class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMistralConfig

    def __init__(self, config, ncls_token_id):
        
        super(MistralForCausalLM, self).__init__(config)
        self.model = LlavaMistralModel(config)
        # self.tokenizer = tokenizer
        self.padding_idx = config.pad_token_id
        
        self.config = config 
        self.ncls_token_id = ncls_token_id
        
        #----------------------------------------------------------#
        if hasattr(config, "sparse_config") and config.sparse_config is not None:
            self.ncls_count = config.sparse_config["ncls_count"]  
            self.hidden_dim = config.sparse_config["hidden_dim"]
            self.output_dim = config.sparse_config["output_dim"]
            self.mlp_type = config.sparse_config["mlp_type"]
            self.loss_threshold = config.sparse_config["loss_threshold"]
            self.temperature = config.sparse_config["temperature"]
            self.use_local_loss = config.sparse_config["use_local_loss"]
            self.feature_layer = config.sparse_config["feature_layer"]
            self.special_tokens_mlp_type = config.sparse_config["special_tokens_mlp_type"]
            self.use_ca_loss = config.sparse_config["use_ca_loss"]
            self.inference_type = config.sparse_config["inference_type"]
       
        #----------------------------------------------------------#
        if self.special_tokens_mlp_type == 1:
            self.special_token_mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.GELU(),
                nn.Linear(config.hidden_size // 4, config.hidden_size)
            )
        elif self.special_tokens_mlp_type == 2:
            self.special_token_mlp = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Dropout(0.3),
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.GELU(),
                nn.Linear(config.hidden_size // 4, config.hidden_size)
            )
     
        self.cross_attention_module = CrossAttentionModule(hidden_size=config.hidden_size)
        # self.logit_scale = nn.Parameter(
        #     torch.tensor(0.1)
        # )
        self.mis_mlp = mis_mlp(input_dim = config.hidden_size, hidden_dim = self.hidden_dim, output_dim = self.output_dim, mlp_type = self.mlp_type)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.cross_attention = None
        # Initialize weights and apply final processing
        self.post_init()
    
    def initialize_mis_mlp(self):
        """确保 mis_mlp 的初始化在模型权重加载后完成"""
        
        self.mis_mlp = mis_mlp(input_dim = self.config.hidden_size, hidden_dim = self.hidden_dim, output_dim = self.output_dim, mlp_type = self.mlp_type)
        self.cross_attention_module = CrossAttentionModule(hidden_size = self.config.hidden_size)
        
        if self.special_tokens_mlp_type == 1:
            self.special_token_mlp = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size // 4),
                nn.GELU(),
                nn.Linear(self.config.hidden_size // 4, self.config.hidden_size)
            )
        elif self.special_tokens_mlp_type == 2:
            self.special_token_mlp = nn.Sequential(
                nn.LayerNorm(self.config.hidden_size),
                nn.Dropout(0.3),
                nn.Linear(self.config.hidden_size, self.config.hidden_size // 4),
                nn.GELU(),
                nn.Linear(self.config.hidden_size // 4, self.config.hidden_size)
            )
            
        # 注册到模块列表中
        self.add_module("mis_mlp", self.mis_mlp)
        self.add_module("special_token_mlp", self.special_token_mlp)
        self.add_module("cross_attention_module", self.cross_attention_module)
        
        for param in self.mis_mlp.parameters():
            param.requires_grad = True
        for param in self.special_token_mlp.parameters():
            param.requires_grad = True
        for param in self.cross_attention_module.parameters():
            param.requires_grad = True
        
    def update_tensor(self, batch_size, tensor, ncls_tensor):
    # 直接在序列末尾追加 ncls 标记
        updated_tensors = []
        for i in range(batch_size):
            # 将 tensor[i] 和 ncls_tensor[i] 在 dim=0 维度上拼接
            updated_tensor = torch.cat([tensor[i], ncls_tensor[i]], dim=0)
            # 将拼接后的 tensor 添加到 updated_tensors 列表中
            updated_tensors.append(updated_tensor)
        # 将 updated_tensors 列表中的 tensor 按顺序堆叠起来
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
        
        # 推理时处理所有类别矩阵
        if return_emb:
            txt_input_ids = input_ids.clone() 
            txt_attention_mask = attention_mask.clone()
            txt_past_key_values = past_key_values
        else:
            txt_input_ids = category_ids.squeeze(1)
            txt_attention_mask = category_attention_mask.squeeze(1)
            txt_past_key_values = past_key_values
        
        # 如果仅有图像输入，则使用图像的多模态预处理
        if inputs_embeds is None and images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                images=images,
                image_sizes=image_sizes
            )

        # 如果仅有类别输入，则处理类别文本的嵌入
        if txt_input_ids is not None:
            (
                txt_input_ids,
                txt_position_ids,
                txt_attention_mask,
                txt_past_key_values,
                txt_inputs_embeds
            ) = self.prepare_inputs_txt_labels_for_multimodal(
                input_ids=txt_input_ids,
                position_ids=txt_position_ids,  
                attention_mask=txt_attention_mask,
                past_key_values=txt_past_key_values
            )

        # 添加 ncls 特殊标记逻辑到 inputs_embeds 和 txt_inputs_embeds
        if inputs_embeds is not None:
            ncls_embeddings = self.special_token_mlp(
                torch.randn(
                    inputs_embeds.size(0),
                    self.ncls_count,
                    inputs_embeds.size(-1),
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype
                )
            )
            ncls_attention_mask = torch.ones(
                (inputs_embeds.size(0), self.ncls_count), dtype=attention_mask.dtype, device=attention_mask.device
            )

            # 更新 inputs_embeds 和 attention_mask
            inputs_embeds = self.update_tensor(inputs_embeds.size(0), inputs_embeds, ncls_embeddings)
            attention_mask = self.update_tensor(inputs_embeds.size(0), attention_mask, ncls_attention_mask)

        if txt_inputs_embeds is not None:
            category_ncls_embeddings = self.special_token_mlp(
                torch.randn(
                    txt_inputs_embeds.size(0),
                    self.ncls_count,
                    txt_inputs_embeds.size(-1),
                    device=txt_inputs_embeds.device,
                    dtype=txt_inputs_embeds.dtype
                )
            )
            category_ncls_attention_mask = torch.ones(
                (txt_inputs_embeds.size(0), self.ncls_count), dtype=txt_attention_mask.dtype, device=txt_attention_mask.device
            )

            # 更新 txt_inputs_embeds 和 txt_attention_mask
            txt_inputs_embeds = self.update_tensor(txt_inputs_embeds.size(0), txt_inputs_embeds, category_ncls_embeddings)
            txt_attention_mask = self.update_tensor(txt_inputs_embeds.size(0), txt_attention_mask, category_ncls_attention_mask)
        
        
        # 添加特殊标记逻辑
        if position_ids is not None:
            ncls_position_ids = torch.arange(
                position_ids.size(-1), position_ids.size(-1) + self.ncls_count,
                dtype=position_ids.dtype, device=position_ids.device
            ).unsqueeze(0).expand(position_ids.size(0), -1)
            position_ids = self.update_tensor(position_ids.size(0), position_ids, ncls_position_ids)

        if labels is not None:
            ncls_labels = torch.full(
                (labels.size(0), self.ncls_count), fill_value=-100,  # 视情况替换忽略标记
                dtype=labels.dtype, device=labels.device
            )
            labels = self.update_tensor(labels.size(0), labels, ncls_labels)

        
        # 获取模型输出
        outputs = super().forward(
            input_ids=input_ids,  
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            txt_input_ids=txt_input_ids, 
            txt_inputs_embeds=txt_inputs_embeds,
            txt_attention_mask=txt_attention_mask,
            return_emb=return_emb
        )

        return outputs
        # if input_ids is not None and attention_mask is not None:
        #     batch_size = input_ids.size(0)

        #     # 创建 ncls 标记
        #     ncls_token_ids = torch.full((batch_size, self.ncls_count), self.ncls_token_id, dtype=input_ids.dtype, device=input_ids.device)
        #     ncls_attention_mask = torch.ones((batch_size, self.ncls_count), dtype=attention_mask.dtype, device=attention_mask.device)

        #     # 直接在序列末尾添加 ncls 标记
        #     input_ids = self.update_tensor(batch_size, input_ids, ncls_token_ids)
        #     attention_mask = self.update_tensor(batch_size, attention_mask, ncls_attention_mask)

        #     # 更新 position_ids
        #     if position_ids is None:
        #         position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        #     else:
        #         # 添加位置标记到末尾
        #         ncls_position_ids = torch.arange(position_ids.size(1), position_ids.size(1) + self.ncls_count, device=position_ids.device).unsqueeze(0).expand(batch_size, -1)
        #         position_ids = self.update_tensor(batch_size, position_ids, ncls_position_ids)

        #     # 更新 labels
        #     if labels is not None:
        #         ncls_labels = torch.full((batch_size, self.ncls_count), -100, dtype=labels.dtype, device=labels.device)  # -100 用于忽略 ncls 部分
        #         labels = self.update_tensor(batch_size, labels, ncls_labels)

        
        # # 更新类别输入
        # if category_ids is not None and category_attention_mask is not None:
        #     category_valid_lengths = category_attention_mask.sum(dim=1).long()
        #     category_ncls_token_ids = torch.full((batch_size, self.ncls_count), self.ncls_token_id, dtype=category_ids.dtype, device=category_ids.device)
        #     category_ncls_attention_mask = torch.ones((batch_size, self.ncls_count), dtype=category_attention_mask.dtype, device=category_attention_mask.device)
            
        #     # 直接在类别输入的序列末尾添加 ncls 标记
        #     category_ids = self.update_tensor(batch_size, category_ids.squeeze(1), category_ncls_token_ids)
        #     category_attention_mask = self.update_tensor(batch_size, category_attention_mask.squeeze(1), category_ncls_attention_mask)
        
        # # 推理时处理所有类别矩阵
        # if return_emb:
        #     txt_input_ids = input_ids.clone() 
        #     txt_attention_mask = attention_mask.clone()
        #     txt_past_key_values = past_key_values
        # else:
        #     txt_input_ids = category_ids
        #     txt_attention_mask = category_attention_mask
        #     txt_past_key_values = past_key_values
      
            
        # # 如果仅有图像输入，则使用图像的多模态预处理
        # if inputs_embeds is None and images is not None:
        #     input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
        #         input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes
        #     )
        
        # # 如果仅有类别输入，则处理类别文本的嵌入
        # if txt_input_ids is not None:
        #     txt_input_ids, txt_position_ids, txt_attention_mask, txt_past_key_values, txt_inputs_embeds = self.prepare_inputs_txt_labels_for_multimodal(
        #         txt_input_ids, txt_position_ids, txt_attention_mask, txt_past_key_values
        #         )
        
        
        # # 获取模型的文本输出
        # outputs = super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     txt_input_ids=txt_input_ids,
        #     txt_position_ids=txt_position_ids,
        #     txt_attention_mask=txt_attention_mask,
        #     txt_past_key_values=txt_past_key_values,
        #     txt_inputs_embeds=txt_inputs_embeds,
        #     return_emb=return_emb
        # )
        

        # return outputs

    def get_model(self):
        # 假设你返回一个基础模型或者相关的模型实例
        return self.base_model
    
    
    @torch.no_grad()
    def inference_pipeline(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        global_category_embeddings_cache: torch.Tensor,  # 新增参数，用于传递全局类别特征向量
        # local_category_embeddings_cache: torch.Tensor,
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
        global_image_embedding = image_output.hidden_states[-self.feature_layer][:, -self.ncls_count:, :].mean(dim=1)  # 使用平均池化
        # local_image_embedding = image_output.hidden_states[-self.feature_layer][:, :-self.ncls_count, :].mean(dim=1)
        # image_embedding = torch.max(image_embedding, dim=1).values  # 使用最大池化
        
        global_image_embedding = self.mis_mlp(global_image_embedding)
        # local_image_embedding = self.mis_mlp(local_image_embedding)
        
        # 步骤2: 对图像特征和类别特征进行L2归一化
        norm_global_image_embedding = F.normalize(global_image_embedding, p=2, dim=-1)
        # norm_local_image_embedding = F.normalize(local_image_embedding, p=2, dim=-1)
        
        norm_global_category_embeddings_cache = F.normalize(global_category_embeddings_cache, p=2, dim=-1)
        # norm_local_category_embeddings_cache = F.normalize(local_category_embeddings_cache, p=2, dim=-1)
        
        similarity_matrix = torch.matmul(norm_global_image_embedding, norm_global_category_embeddings_cache.T) / self.temperature  # 计算余弦相似度
        # 将相似度矩阵转换为概率分布 
        similarity_probs = similarity_matrix.softmax(dim=-1)
        # if self.inference_type == 1:
        #     # 计算余弦相似度矩阵
        #     similarity_matrix = torch.matmul(norm_global_image_embedding, norm_global_category_embeddings_cache.T) / self.temperature  # 计算余弦相似度
        #     # 将相似度矩阵转换为概率分布 
        #     similarity_probs = similarity_matrix.softmax(dim=-1)
        # elif self.inference_type == 2:
        #     # 图像全局特征作为查询，文本局部特征作为键和值进行注意力计算
        #     global_image_embedding = global_image_embedding.repeat(local_category_embeddings_cache.size(0), 1)  # (seq_len, hidden_size)
        #     image_to_text_features, _ = self.cross_attention_module(global_image_embedding, local_category_embeddings_cache)
        #     # 文本全局特征作为查询，图像局部特征作为键和值进行注意力计算
        #     local_image_embedding = local_image_embedding.repeat(global_category_embeddings_cache.size(0), 1)  # (seq_len, hidden_size)
        #     text_to_image_features, _ = self.cross_attention_module(global_category_embeddings_cache, local_image_embedding)
        #     # 归一化特征向量到单位球面
        #     image_to_text_features = F.normalize(image_to_text_features, p=2, dim=-1)  # (B, 4096)
        #     text_to_image_features = F.normalize(text_to_image_features, p=2, dim=-1)  # (B, 4096)

        #     # Step 1: 图像到文本的 similarity_matrix
        #     similarity_matrix_image_to_text = torch.matmul(image_to_text_features, norm_global_category_embeddings_cache.T) / self.temperature  # (B, B)
    
        #     # Step 2: 文本到图像的 similarity_matrix
        #     similarity_matrix_text_to_image = torch.matmul(text_to_image_features, norm_global_image_embedding.T) / self.temperature  # (B, B)
            
        #     similarity_matrix = (similarity_matrix_image_to_text + similarity_matrix_text_to_image) / 2  # (B, B)
        #     similarity_probs = similarity_matrix.softmax(dim=-1)
        #     print(similarity_probs)

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
