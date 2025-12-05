from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return x + out * 0.1


class SRNet(nn.Module):
    """Very small EDSR-style super-resolution network."""

    def __init__(self, scale: int = 2, num_feats: int = 64, num_blocks: int = 8):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, num_feats, 3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(num_feats) for _ in range(num_blocks)],
            nn.Conv2d(num_feats, num_feats, 3, padding=1),
        )

        up_layers = []
        for _ in range(int(math.log2(scale))):
            up_layers += [
                nn.Conv2d(num_feats, num_feats * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ]
        self.upsampler = nn.Sequential(*up_layers)
        self.tail = nn.Conv2d(num_feats, 3, 3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        res = self.body(feat)
        feat = feat + res
        up = self.upsampler(feat)
        out = self.tail(up)
        return torch.sigmoid(out)


class DnCNN(nn.Module):
    """Lightweight DnCNN-style denoiser."""

    def __init__(self, depth: int = 10, num_feats: int = 64, in_channels: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_channels, num_feats, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(num_feats, num_feats, 3, padding=1),
                nn.BatchNorm2d(num_feats),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(num_feats, in_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.net(x)
        return torch.clamp(x - noise, 0.0, 1.0)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ColorizationUNet(nn.Module):
    """Predicts ab channels from a normalized L input."""

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.down1 = DoubleConv(1, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 2, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.conv2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.conv1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 2, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        u2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv2(u2)

        u1 = self.up1(c2)
        u1 = torch.cat([u1, d1], dim=1)
        c1 = self.conv1(u1)

        out = self.out_conv(c1)
        return torch.tanh(out)


class InpaintUNet(nn.Module):
    """Reconstructs missing pixels using the masked RGB image and binary mask."""

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.down1 = DoubleConv(4, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 2, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.conv2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.conv1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 3, 1)

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=1)

        d1 = self.down1(inp)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        u2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv2(u2)

        u1 = self.up1(c2)
        u1 = torch.cat([u1, d1], dim=1)
        c1 = self.conv1(u1)

        out = self.out_conv(c1)
        return torch.clamp(out, 0.0, 1.0)


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "super_resolution": {
        "display_name": "Super Resolution (x2)",
        "constructor": SRNet,
        "init_kwargs": {"scale": 2},
        "checkpoint": ROOT_DIR / "experiments/2025-11-23_22-00-48/sr2/best.pth",
        "description": "Upscale low-resolution scans while keeping details sharp.",
    },
    "denoise": {
        "display_name": "Denoising",
        "constructor": DnCNN,
        "init_kwargs": {},
        "checkpoint": ROOT_DIR / "experiments/2025-11-23_23-26-16/denoise/best.pth",
        "description": "Remove Gaussian noise and grain introduced by aging.",
    },
    "colorize": {
        "display_name": "Colorization",
        "constructor": ColorizationUNet,
        "init_kwargs": {},
        "checkpoint": ROOT_DIR / "experiments/2025-11-24_01-37-27/colorize/best.pth",
        "description": "Bring grayscale photos back to life with plausible colors.",
    },
    "inpaint": {
        "display_name": "Inpainting",
        "constructor": InpaintUNet,
        "init_kwargs": {},
        "checkpoint": ROOT_DIR / "experiments/2025-11-24_02-51-06/inpaint/best.pth",
        "description": "Fill scratches, tears, or missing regions using context-aware completion.",
    },
}
