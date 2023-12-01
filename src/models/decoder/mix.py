import torch
import torch.nn as nn


class MixedDecoder(nn.Module):
    def __init__(self, decoders):
        super().__init__()
        self.decoders = nn.ModuleList(decoders)


    def forward(self, x):
        out = []
        for decoder in self.decoders:
            out.append(decoder(x))
        ret = torch.mean(torch.stack(out), dim=0)
        return ret
    
if __name__ == "__main__":
    channels = 8
    num_classes = 3
    num_timesteps = 1440
    decoder = MixedDecoder([
        nn.Linear(channels, num_classes),
        nn.Linear(channels, num_classes),
        nn.Linear(channels, num_classes),
    ])
    x = torch.rand(2, num_timesteps, channels)
    y = decoder(x)
    print(y.shape)