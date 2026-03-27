""" ConvMixer

Downstripped Hugging face implementation from here:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convmixer.py
"""
import math
import torch
import torch.nn as nn
from typing import Tuple, Union


class SimpleTransformer(nn.Module):
    def __init__(self, methods: int = 6, lag_in: int = 4, lag_out: int = 4, n_vars: int = 7, 
                 model_dim: int = 256, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1, pos_embedding_dim: int = 16):
        super(SimpleTransformer, self).__init__()
        
        self.methods = methods
        self.lag_in = lag_in
        self.lag_out = lag_out
        self.n_vars = n_vars
        self.model_dim = model_dim
        
        # Calculate Input Dimension
        input_total_dim = lag_in * n_vars * n_vars
        output_total_dim = lag_out * n_vars * n_vars
        
        
        # We need to project the flattened input (vars*vars*lags) for each method into `model_dim`
        # Since we are concatenating the positional embedding, we reduce the input projection output size
        self.pos_embedding_dim = pos_embedding_dim
        self.input_model_dim = model_dim - pos_embedding_dim

        assert self.input_model_dim > input_total_dim, "Please choose a higher model dim. Information is destroyed in the first layer"

        self.input_proj = nn.Linear(input_total_dim, self.input_model_dim)
        
        # Learnable CLS token - uses the full model_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        
        # Positional encoding for methods
        # This will be concatenated to the input
        self.pos_encoder = nn.Parameter(torch.randn(1, methods, pos_embedding_dim))
        
        # Dropout and Norm for the embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        # We can pass a final normalization layer to the encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim))
        
        # Output projection - we project the aggregated representation back to vars*vars*lags
        self.output_proj = nn.Linear(model_dim, output_total_dim)

    def forward(self, x): 
        # The input shape should be (batch, methods, vars* vars* lags)
        # We use the methods as the sequence
        b = x.size(0)
        
        # Flatten the last three dimensions: vars, vars, lags -> input_total_dim
        x = x.view(b, self.methods, -1) # -> (batch, methods, input_total_dim)
        # Transformation to the embedding dimension (model_dim - embedding_dim)
        x = self.input_proj(x) # -> (batch, methods, model_dim - embedding_dim)
        
        # Expand positional encoding to batch size
        pos_enc = self.pos_encoder.expand(b, -1, -1)
        
        # Concatenate input projection and positional encoding along the feature dimension (dim=-1)
        x = torch.cat((x, pos_enc), dim=-1) # -> (batch, methods, model_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # -> (batch, methods + 1, model_dim)

        # Apply Norm and Dropout to the full embedding before the transformer
        x = self.input_norm(x)
        x = self.embedding_dropout(x)

        # Transformer encoder (batch_first=True)
        x = self.transformer_encoder(x) # -> (batch, methods + 1, model_dim)
        
        # Use the CLS token for prediction
        x = x[:, 0] # -> (batch, model_dim)
        # Project back to output size
        x = self.output_proj(x) # -> (batch, input_total_dim)
        
        # Reshape to (batch, vars, vars, lags)
        x = x.reshape(b, self.n_vars, self.n_vars, self.lag_out)
        return x
