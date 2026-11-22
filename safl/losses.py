"""Conventional segmentation and classification losses retained from the workflow."""
import torch
from torch import nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, positive_weight=2.0, dice_weight=0.8):
        super().__init__()
        self.register_buffer("positive_weight", torch.tensor([positive_weight]))
        self.dice_weight = dice_weight

    def forward(self, output, target):
        logits = output["low_res_logits"].float()
        target = target.float()
        if logits.shape != target.shape:
            raise ValueError("Segmentation logits and low-resolution masks must have equal shapes")
        ce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.positive_weight)
        probability = torch.sigmoid(logits)[:, 0]
        truth = target[:, 0]
        dice = 1 - (2 * (probability * truth).sum() + 1e-5) / (
            probability.square().sum() + truth.square().sum() + 1e-5)
        return (1 - self.dice_weight) * ce + self.dice_weight * dice


class ClassificationLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(logits.float(), target.long(), reduction="none",
                             label_smoothing=self.label_smoothing)
        return (self.alpha * (1 - torch.exp(-ce)).pow(self.gamma) * ce).mean()


def joint_loss(output, logits, masks, labels, segmentation, classification):
    seg = segmentation(output, masks)
    cls = classification(logits, labels)
    return seg + cls, seg, cls
