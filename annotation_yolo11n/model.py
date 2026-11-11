"""R4 architecture: YOLO11n feature pyramid + P2 detail + semantic spatial points."""
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO


class HeatmapModel(nn.Module):
    def __init__(self, num_classes=2, backbone=None):
        super().__init__()
        # YAML builds the architecture locally without downloading weights when loading a saved state.
        yolo = YOLO(str(backbone) if backbone else "yolo11n-pose.yaml").model
        self.layers = nn.ModuleList(list(yolo.model)[:23])
        self.saved = set(yolo.save) | {2}
        for parameter in self.layers.parameters():
            parameter.requires_grad_(True)
        self.p2 = nn.Sequential(nn.Conv2d(64, 32, 1), nn.GroupNorm(8, 32), nn.SiLU())
        self.semantic = nn.Sequential(nn.Conv2d(448, 64, 1), nn.GroupNorm(8, 64), nn.SiLU())
        self.fuse = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.heatmap = nn.Conv2d(64, num_classes, 1)
        self.visibility = nn.Sequential(nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, num_classes))

    def forward(self, x):
        stored, features = [], {}
        for index, layer in enumerate(self.layers):
            if layer.f != -1:
                x = stored[layer.f] if isinstance(layer.f, int) else [x if j == -1 else stored[j] for j in layer.f]
            x = layer(x)
            stored.append(x if index in self.saved else None)
            if index in [2, 16, 19, 22]:
                features[index] = x
        semantic = self.semantic(torch.cat([
            F.interpolate(features[i], size=features[16].shape[-2:], mode="bilinear", align_corners=False)
            for i in [16, 19, 22]
        ], dim=1))
        z = self.fuse(torch.cat([
            self.p2(features[2]),
            F.interpolate(semantic, size=features[2].shape[-2:], mode="bilinear", align_corners=False),
        ], dim=1))
        return self.heatmap(z), self.visibility(z.mean((-2, -1)))


def decode(logits, visibility_logits, original_size=(640, 480)):
    """Peak-local5x5 soft centroid. original_size is (width,height), not (height,width)."""
    height, width = logits.shape[-2:]
    indices = logits.float().flatten(2).argmax(-1)
    px, py = (indices % width).float(), (indices // width).float()
    yy, xx = torch.meshgrid(torch.arange(height, device=logits.device),
                            torch.arange(width, device=logits.device), indexing="ij")
    mask = ((xx[None, None] - px[:, :, None, None]).abs() <= 2)
    mask = mask & ((yy[None, None] - py[:, :, None, None]).abs() <= 2)
    probability = logits.float().masked_fill(~mask, -1e4).flatten(2).softmax(-1).view_as(logits)
    x = (probability * xx).sum((-2, -1)) * (original_size[0] / width)
    y = (probability * yy).sum((-2, -1)) * (original_size[1] / height)
    return torch.stack([x, y], dim=-1), visibility_logits.float().sigmoid()
