import torch
import torch.nn.functional as F
from torch import nn


class CostSensitiveLoss(nn.Module):
    def __init__(self, cost_matrix):
        super(CostSensitiveLoss, self).__init__()
        self.cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32)

    def forward(self, outputs, labels):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(outputs, labels)

        # 计算每个样本的预测标签
        _, preds = torch.max(outputs, 1)

        # 计算混淆矩阵损失
        cost_loss = 0.0
        for i in range(len(labels)):
            true_class = labels[i].item()
            pred_class = preds[i].item()
            cost_loss += self.cost_matrix[true_class, pred_class]

        # 返回综合损失
        total_loss = ce_loss + (cost_loss / len(labels))  # 归一化成本损失
        return total_loss
