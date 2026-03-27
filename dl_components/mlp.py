""" ConvMixer

Downstripped Hugging face implementation from here:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convmixer.py
"""
import torch.nn as nn
from typing import Tuple, Union


class SimpleMLP(nn.Module):
    def __init__(self, methods= 6, lag_in=4, lag_out=3, n_vars=7, hidden_dims:list[int] = [32, 64], dropout_rate: float = 0.0, use_batch_norm: bool = True):
        super(SimpleMLP, self).__init__()

        input_dim = methods * lag_in * n_vars * n_vars
        output_dim = lag_out * n_vars * n_vars
        self.n_vars = n_vars
        self.lag_out_dimension = lag_out
        layers = []
        
        # Input layer
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor except the batch dimension        
        x =  self.network(x)
        x = x.reshape(x.size(0), self.n_vars, self.n_vars,self.lag_out_dimension)
        return x