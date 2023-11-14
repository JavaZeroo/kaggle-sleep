import torch
import torch.nn as nn
import math

class TransformerFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels,
        height,
        num_layers,
        nhead,
        dim_feedforward,
        out_size
    ):
        super().__init__()
        self.height = height
        self.fc = nn.Linear(in_channels, height)
        self.positional_encoding = PositionalEncoding(height, max_len=out_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=height,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers,
        )
        self.out_chans = 1
        self.out_size = out_size
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_size is not None:
            x = x.unsqueeze(1)
            x = self.pool(x)
            x = x.squeeze(1)
            
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x    
