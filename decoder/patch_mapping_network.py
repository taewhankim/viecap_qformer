import torch
import math
from torch import nn
from PIL import Image
from PUREcap.modeling_qformer import Patch_mapping
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from decoder.deconv_utils import noise_injection

import torch.nn.functional as F

class Upsampling(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.embed_dim = config.embed_dim ## 640
        self.hidden_size = config.hidden_size ## 768
        self.patch_num = config.patch_num

        self.k = self.config.k
        self.noise_variance = self.config.noise_var
        self.PM = Patch_mapping(self.config)

        self.template_position = nn.Parameter(torch.randn(1, self.config.sequence_length, self.hidden_size))

        self.unproj = nn.Linear(self.embed_dim,self.hidden_size)

    def forward(self, x, ori, noise_ori=None):
        '''
        x = clip text feat
        ori = noise + text feat -> unproj -> image hidden feat 처럼
        '''
        ### repeat
        # x = x.unsqueeze(1)
        # x = x.repeat(1,self.patch_num,1) #200
        x  = noise_injection(x, variance=self.noise_variance)

        x = self.unproj(x)
        x_norm = x / x.norm(dim=-1, keepdim=True)
        # prefix_norm = ori/ori.norm(dim=-1, keepdim=True)
        # x_norm = x/x.norm(dim=-1, keepdim=True)
        # result = torch.einsum('bik,bkj->bij', prefix_norm, x_norm.permute(0,2,1))
        # _, index = torch.max(result, dim=-1) ## 64, 50, 1
        #
        # retrieved_rows = x[torch.arange(x.size(0))[:, None], index]
        #
        # retrieved_rows = retrieved_rows + self.template_position[:,:retrieved_rows.size(1),:]
        #
        # retrieved_rows_norm = retrieved_rows/retrieved_rows.norm(dim=-1, keepdim=True)

        output = self.PM(query_tokens = ori,
                         condition_embeds = x_norm)

        # output = output + x_norm


        return output


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=1):
        super().__init__()

        # compute positional encoding in log space
        pe = torch.zeros(max_len, d_model).float()
        pe.required_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # return self.pe[:, :x.size(1)]
        return self.pe[:x.size(1), :]
