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
            logits: tensor of shape (N, C, H, W)
            label: tensor of shape(N, C, H, W)
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
            logits: tensor of shape (N, C, H, W)
            label: tensor of shape(N, C, H, W)
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


class BCELogCoshDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BCELogCoshDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, target):
        """
        inputs:
            logits: tensor of shape (N, C, H, W)
            label: tensor of shape(N, C, H, W)
        output:
            loss: tensor of shape(1, )
        """
        preds = torch.sigmoid(preds)
        intersection = (preds * target).sum(dim=(2, 3))
        union = (preds + target).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        log_cosh_dice = torch.log(torch.cosh(1 - dice))
        bce = loss_bce(preds, target)
        loss = torch.mean(log_cosh_dice) + bce
        return loss
