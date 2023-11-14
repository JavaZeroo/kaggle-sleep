import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, loss1, loss2):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2

    def forward(self, predictions, targets):
        # 假设 predictions 和 targets 都是形状 (batch_size, n_timesteps, 3)
        # 第一个维度使用 BCE 损失
        loss1 = self.loss1(predictions[:, :, 0], targets[:, :, 0])

        # 后两个维度使用 MSE 损失
        loss2 = self.loss2(predictions[:, :, 1:], targets[:, :, 1:])

        # 结合两种损失
        combined_loss = (loss1 + loss2) / 2

        return combined_loss