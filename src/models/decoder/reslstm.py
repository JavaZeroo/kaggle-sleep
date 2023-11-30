import torch
import torch.nn as nn


class ResLSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        n_classes: int,
        nhead: int=8,  # 添加多头注意力的头数
    ):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList()

        self.act = nn.SELU()        
        
        lstm_out_hidden = hidden_size * 2 if bidirectional else hidden_size
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=nhead,
            dropout=dropout,
            batch_first=True  # 确保batch维度在前
        )

        for i in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    dropout=0,
                    bidirectional=bidirectional,
                    batch_first=True,
                ),
                nn.LayerNorm(lstm_out_hidden),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_hidden, hidden_size),
            ]))
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch_size, n_timesteps, n_channels)
        x = self.fc_in(x)
        x = self.ln(x)
        h = None
        for layer in self.layers:
            lstm, ln, dropout, linear = layer
            tx, h = lstm(x, h)
            tx = ln(tx)
            tx = linear(tx)
            tx = self.act(tx)
            
            attn_output, _ = self.multihead_attn(tx, tx, tx)  # 应用多头注意力
            tx = attn_output + tx  # 添加残差连接

            # tx = dropout(tx)
            x = x + tx

        x = self.linear(x)
        return x

if __name__ == "__main__":
    
    channels = 8
    
    decoder = ResLSTMDecoder(
        input_size=channels,
        hidden_size=64,
        num_layers=4,
        dropout=0.2,
        bidirectional=True,
        n_classes=3,
        nhead=4,
        )
    
    print("Total parameters:", sum(p.numel() for p in decoder.parameters())/1e6, "M")
    
    x = torch.rand(1, channels, 1440)

    print(decoder(x).shape)