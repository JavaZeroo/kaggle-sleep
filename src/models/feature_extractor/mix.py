import torch
import torch.nn as nn

class MixedFeatureExtractor(nn.Module):
    def __init__(self, extractors):
        super().__init__()
        self.extractors = nn.ModuleList(extractors)
        # make sure all extractors have the same height
        assert len(set([extractor.height for extractor in extractors])) == 1 
        
        self.height = self.extractors[0].height
        self.out_chans = sum([extractor.out_chans for extractor in extractors])
        

    def forward(self, x):
        
        outs = [extractor(x) for extractor in self.extractors]
        combined = torch.cat(outs, dim=1)
        return combined