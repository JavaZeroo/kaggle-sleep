import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
if __name__ == '__main__':
    loss_fn = FocalLoss(gamma=2)
    # logits假设是未经sigmoid的原始输出
    input = torch.rand(3, 1440, 3)
    # 目标现在是0或1
    target = torch.rand(3, 1440, 3)
    print(target.min(), target.max())
    loss = loss_fn(input, target)
    print(loss)
