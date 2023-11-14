import torch
import torch.nn as nn

class MixedFeatureExtractor(nn.Module):
    def __init__(self, extractor1, extractor2, ):
        super().__init__()
        self.extractor1 = extractor1
        self.extractor2 = extractor2
        assert extractor1.height == extractor2.height
        self.height = extractor1.height
        self.out_chans = extractor1.out_chans + extractor2.out_chans
        

    def forward(self, x):
        extractor1_out = self.extractor1(x)
        extractor2_out = self.extractor2(x)
        # print(extractor1_out.shape, extractor2_out.shape)
        combined = torch.cat([extractor1_out, extractor2_out], dim=1)
        # output = self.fc(combined)
        return combined