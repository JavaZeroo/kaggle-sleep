import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(n_channels, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = x.transpose(1, 2)
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.selu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = F.selu(self.fc3(x))
        x = F.dropout(x, p=0.5)
        x = F.selu(self.fc4(x))
        x = F.dropout(x, p=0.5)
        x = self.fc5(x)
        return x
