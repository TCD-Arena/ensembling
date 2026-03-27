""" ConvMixer

Downstripped Hugging face implementation from here:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convmixer.py
"""
import torch.nn as nn
from typing import Tuple, Union


class SimpleLinear(nn.Module):
    def __init__(self, methods= 6, lag_in=4, lag_out=4, n_vars=7):
        super(SimpleLinear, self).__init__()
        
        self.lag_in = lag_in
        self.lag_out = lag_out
        self.n_vars = n_vars
        self.linear = nn.Linear(methods*lag_in*n_vars*n_vars, lag_out*n_vars*n_vars)


    def forward(self, x): 
        
        x = x.view(x.size(0), -1)  # Flatten the input tensor except the batch dimension        
    
        x = self.linear(x)  # Flatten the input tensor
        # reshape to proper output shape
        x = x.reshape(x.size(0), self.n_vars, self.n_vars, self.lag_out)
        return x



