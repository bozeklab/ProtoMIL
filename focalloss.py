""" based on https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289 """
import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha]).cuda()

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none') # ce_loss is a vector of -log(pt)
        pt = torch.exp(-ce_loss) # e^(-log(pt)) = pt
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean() # mean over the batch
        return focal_loss * 2
