import torch
from torch import nn
from PUREcap.qformer import Blip2QFormerModel
from typing import Any, Optional, Tuple, Union

class Patch_mapping(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.hidden_size))
        self.qformer = Blip2QFormerModel(config)
        # self.visual_projection = nn.Linear(config.hidden_size, config.vision_hidden_size)

    def forward(self,
                condition_embeds: torch.FloatTensor,
                query_tokens: torch.FloatTensor,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):

        image_attention_mask = torch.ones(condition_embeds.size()[:-1], dtype=torch.long, device=condition_embeds.device)

        # query_tokens = self.query_tokens.expand(condition_embeds.shape[0], -1, -1)

        self.qformer.dtype = query_tokens.dtype
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=condition_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]
        ### 이 부분은 추가할지 말지 고민
        # language_model_inputs = self.visual_projection(query_output)

        # return language_model_inputs
        return query_output