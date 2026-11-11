"""Visible-only Gaussian spatial CE; known-only balanced visibility BCE."""
import torch
from torch.nn import functional as F


def point_loss(logits, visibility_logits, points, status, sigma=1.5):
    # status: 2=visible,0=confirmed invisible,1=unknown(ignore/pending).
    height, width = logits.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=logits.device),
                            torch.arange(width, device=logits.device), indexing="ij")
    distance = (xx[None, None] - points[:, :, 0, None, None]).square()
    distance = distance + (yy[None, None] - points[:, :, 1, None, None]).square()
    target = torch.exp(-distance / (2 * sigma**2))
    target = target / target.sum((-2, -1), keepdim=True).clamp_min(1e-9)
    ce = -(target * F.log_softmax(logits.float().flatten(2), -1).view_as(logits)).sum((-2, -1))
    positive, negative = status == 2, status == 0
    active = positive.any(0)
    location = (((ce * positive).sum(0) / positive.sum(0).clamp_min(1)) * active).sum()
    location = location / active.sum().clamp_min(1)
    bce = F.binary_cross_entropy_with_logits(visibility_logits.float(), positive.float(), reduction="none")
    pos = (bce * positive).sum(0) / positive.sum(0).clamp_min(1)
    neg = (bce * negative).sum(0) / negative.sum(0).clamp_min(1)
    counts = positive.any(0).float() + negative.any(0).float()
    visibility = ((pos + neg) / counts.clamp_min(1)).sum() / (counts > 0).sum().clamp_min(1)
    return location + visibility
