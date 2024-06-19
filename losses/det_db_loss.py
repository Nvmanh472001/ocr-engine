import torch
import torch.nn.functional as F
from torch import nn


class BalanceLoss(nn.Module):
    def __init__(
        self,
        balance_loss=True,
        main_loss_type="DiceLoss",
        negative_ratio=3,
        return_origin=False,
        eps=1e-6,
        **kwargs,
    ):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            balance_loss (bool): whether balance loss or not, default is True
            main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss',
                'Euclidean','BCELoss', 'MaskL1Loss'], default is  'DiceLoss'.
            negative_ratio (int|float): float, default is 3.
            return_origin (bool): whether return unbalanced loss or not, default is False.
            eps (float): default is 1e-6.
        """
        super(BalanceLoss, self).__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps

        if self.main_loss_type == "CrossEntropy":
            self.loss = nn.CrossEntropyLoss()
        elif self.main_loss_type == "Euclidean":
            self.loss = nn.MSELoss()
        elif self.main_loss_type == "DiceLoss":
            self.loss = DiceLoss(self.eps)
        elif self.main_loss_type == "BCELoss":
            self.loss = BCELoss(reduction="none")
        elif self.main_loss_type == "MaskL1Loss":
            self.loss = MaskL1Loss(self.eps)
        else:
            loss_type = ["CrossEntropy", "DiceLoss", "Euclidean", "BCELoss", "MaskL1Loss"]
            raise Exception("main_loss_type in BalanceLoss() can only be one of {}".format(loss_type))

    def forward(self, pred, gt, mask=None):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            pred (variable): predicted feature maps.
            gt (variable): ground truth feature maps.
            mask (variable): masked maps.
        return: (variable) balanced loss
        """
        positive = gt * mask
        negative = (1 - gt) * mask

        positive_count = int(positive.sum())
        negative_count = int(min(negative.sum(), positive_count * self.negative_ratio))
        loss = self.loss(pred, gt, mask=mask)

        if not self.balance_loss:
            return loss

        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = torch.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            sort_loss, _ = negative_loss.sort(descending=True)
            negative_loss = sort_loss[:negative_count]
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        else:
            balance_loss = positive_loss.sum() / (positive_count + self.eps)
        if self.return_origin:
            return balance_loss, loss

        return balance_loss


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask, weights=None):
        """
        DiceLoss function.
        """

        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = torch.sum(pred * gt * mask)

        union = torch.sum(pred * mask) + torch.sum(gt * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        Mask L1 Loss
        """
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        loss = torch.mean(loss)
        return loss


class BCELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None):
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss


class DBLoss(nn.Module):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        param (dict): the super paramter for DB Loss
    """

    def __init__(
        self,
        balance_loss=True,
        main_loss_type="DiceLoss",
        alpha=5,
        beta=10,
        ohem_ratio=3,
        eps=1e-6,
        **kwargs,
    ):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceLoss(balance_loss=balance_loss, main_loss_type=main_loss_type, negative_ratio=ohem_ratio)

    def forward(self, predicts, labels):
        predict_maps = predicts["maps"]
        label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = labels[1:]
        shrink_maps = predict_maps[:, 0, :, :]
        threshold_maps = predict_maps[:, 1, :, :]
        binary_maps = predict_maps[:, 2, :, :]

        loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map, label_shrink_mask)
        loss_threshold_maps = self.l1_loss(threshold_maps, label_threshold_map, label_threshold_mask)
        loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map, label_shrink_mask)
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps
        # CBN loss
        if "distance_maps" in predicts.keys():
            distance_maps = predicts["distance_maps"]
            cbn_maps = predicts["cbn_maps"]
            cbn_loss = self.bce_loss(cbn_maps[:, 0, :, :], label_shrink_map, label_shrink_mask)
        else:
            dis_loss = torch.tensor([0.0], device=shrink_maps.device)
            cbn_loss = torch.tensor([0.0], device=shrink_maps.device)

        loss_all = loss_shrink_maps + loss_threshold_maps + loss_binary_maps
        losses = {
            "loss": loss_all + cbn_loss,
            "loss_shrink_maps": loss_shrink_maps,
            "loss_threshold_maps": loss_threshold_maps,
            "loss_binary_maps": loss_binary_maps,
            "loss_cbn": cbn_loss,
        }
        return losses
