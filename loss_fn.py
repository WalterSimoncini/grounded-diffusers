import torch
import torch.nn as nn


loss_bce = nn.BCEWithLogitsLoss()


class DiceLoss(nn.Module):
    def __init__(self, p=2, smooth=1):
        super(DiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, preds, target):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """

        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        probs = torch.sigmoid(preds)
        numer = (probs * target).sum()
        denor = (probs.pow(self.p) + target.pow(self.p)).sum()
        loss = 1.0 - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss


class BCEDiceLoss(nn.Module):
    def __init__(self, p=2, smooth=1):
        super(BCEDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, preds, target):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        preds = preds.contiguous().view(preds.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        probs = torch.sigmoid(preds)
        numer = (probs * target).sum()
        denor = (probs.pow(self.p) + target.pow(self.p)).sum()
        dice_loss = 1.0 - (2 * numer + self.smooth) / (denor + self.smooth)
        bce = loss_bce(preds, target)
        loss = dice_loss + bce
        return loss
