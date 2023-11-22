import torch
import torch.nn as nn
from .transformer import TransformerFeatureExtractor

class MixedFeatureExtractor(nn.Module):
    def __init__(self, extractors, transformer_weight=False, attention=False):
        super().__init__()
        self.extractors = nn.ModuleList(extractors)
        assert len(set([extractor.height for extractor in extractors])) == 1 
        
        self.height = self.extractors[0].height
        self.out_chans = sum([extractor.out_chans for extractor in extractors])
        self.transformer_weight = transformer_weight
        self.attention = attention

        # Attention modules
        if self.attention:
            # 确保每个注意力模块的输入维度与对应提取器的输出通道数匹配
            self.attention_modules = nn.ModuleList(
                [nn.Linear(extractor.out_chans, 1) for extractor in extractors]
            )
        
        if self.transformer_weight:
            has_transformer = False
            for extractor in extractors:
                if isinstance(extractor, TransformerFeatureExtractor):
                    has_transformer = True
                    self.out_chans -= extractor.out_chans
                    break
            assert has_transformer, "No transformer extractor found"

    def forward(self, x):
        outs = []
        transformer_ft = None

        if self.transformer_weight:
            for extractor in self.extractors:
                if isinstance(extractor, TransformerFeatureExtractor):
                    transformer_ft = extractor(x)
                else:
                    outs.append(extractor(x))
            combined = torch.cat(outs, dim=1) * transformer_ft
        else:
            for i, extractor in enumerate(self.extractors):
                extracted_features = extractor(x)
                if self.attention:
                        # 全局平均池化
                        gap = torch.mean(extracted_features, dim=[2, 3])
                        attention_weight = torch.sigmoid(self.attention_modules[i](gap)).unsqueeze(2).unsqueeze(3)
                        extracted_features = extracted_features * attention_weight
                outs.append(extracted_features)
            combined = torch.cat(outs, dim=1)
        return combined
