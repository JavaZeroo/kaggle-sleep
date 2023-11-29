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
    ):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList()
        
        lstm_out_hidden = hidden_size * 2 if bidirectional else hidden_size
        
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
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = x.transpose(1, 2)  # (batch_size, n_timesteps, n_channels)
        x = self.fc_in(x)
        x = self.ln(x)
        h = None
        for layer in self.layers:
            lstm, ln, dropout, linear = layer
            tx, h = lstm(x, h)
            tx = ln(tx)
            tx = linear(tx)
            x = x + tx
            x = dropout(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    decoder = ResLSTMDecoder(
        input_size=7,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        n_classes=3,
        )
    
    x = torch.rand(1, 7, 1440)

    print(decoder(x).shape)