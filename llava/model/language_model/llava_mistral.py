from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from llava.model.modeling_mistral import (
    MistralConfig,
    MistralModel,
    MistralForCausalLM
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaMistralConfig(MistralConfig):
    model_type = "llava_mistral"


class LlavaMistralModel(LlavaMetaModel, MistralModel):
    config_class = LlavaMistralConfig

    def __init__(self, config: MistralConfig):
        super(LlavaMistralModel, self).__init__(config)


class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = LlavaMistralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
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
            return_dict=True
        )
        global_image_embedding = image_output.hidden_states[-2].mean(dim=1)
        # local_image_embedding = image_output.hidden_states[-self.feature_layer][:, :-self.ncls_count, :].mean(dim=1)
        
        # 步骤2: 对图像特征和类别特征进行L2归一化
        norm_global_image_embedding = F.normalize(global_image_embedding, p=2, dim=-1)
        # norm_local_image_embedding = F.normalize(local_image_embedding, p=2, dim=-1)
        
        norm_global_category_embeddings_cache = F.normalize(global_category_embeddings_cache, p=2, dim=-1)
        # norm_local_category_embeddings_cache = F.normalize(local_category_embeddings_cache, p=2, dim=-1)
        
        similarity_matrix = torch.matmul(norm_global_image_embedding, norm_global_category_embeddings_cache.T) / 0.05  # 计算余弦相似度
        # 将相似度矩阵转换为概率分布 
        similarity_probs = similarity_matrix.softmax(dim=-1)
      
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
