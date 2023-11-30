import torch
import torch.nn as nn


class enLSTMDecoder(nn.Module):
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
        # self.act = nn.SELU()        

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        hidden_size = hidden_size * 2 if bidirectional else hidden_size
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
        # x = self.act(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    
    channels = 8
    
    decoder = enLSTMDecoder(
        input_size=channels,
        hidden_size=64,
        num_layers=4,
        dropout=0.2,
        bidirectional=True,
        n_classes=3,
        )
    print(decoder.lstm)
    print("Total parameters:", sum(p.numel() for p in decoder.parameters())/1e6, "M")
    
    x = torch.rand(1, channels, 1440)

    print(decoder(x).shape)