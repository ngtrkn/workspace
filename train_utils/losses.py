import torch.nn as nn
import torch

BCE = nn.BCELoss(reduction="none")
EPS = 1e-10
def bce_loss(input_, target, ignore_index=-100, reduction='mean'):
    iflat = input_[target != ignore_index]
    tflat = target[target != ignore_index]
    if len(iflat) == 0: return 0

    out = BCE(iflat, tflat)
    # out = out[target != ignore_index]
    
    if reduction == "mean":
        return torch.mean(out)
    elif reduction == "sum":
        return torch.sum(out)
    else:
        raise ValueError(f"reduction type does not support {reduction}")

def dice_loss(input_, target, ignore_index=-100):
    smooth = 1.0
    iflat = input_[target != ignore_index]
    tflat = target[target != ignore_index]
    if len(iflat) == 0: return 0

    intersection = (iflat * tflat).sum()

    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def contour_loss(y, target=None):
    out = torch.sum(y*(1.-y))
    norm_factor = 1
    for dim in y.shape:
        norm_factor *= dim

    return out / norm_factor