"""Model families requested by the advisor.

1. CNN (ImageNet-pretrained, fully fine-tuned): resnet50, efficientnet_b0, convnext_tiny
2. transfer learning variants (frozen backbone etc.) - TODO
3. vit          - ViT trained from scratch
4. hybrid       - CNN stem + Transformer encoder
5. vision_mamba - bidirectional selective-SSM (Vision Mamba style), pure PyTorch
                  (no mamba-ssm / CUDA kernels needed, so it runs on Windows; slower)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _pretrained(builder, weights):
    return builder(weights=weights.DEFAULT)


def resnet50(nc, **_):
    tv = torchvision.models
    m = _pretrained(tv.resnet50, tv.ResNet50_Weights)
    m.fc = nn.Linear(m.fc.in_features, nc)
    return m


def efficientnet_b0(nc, **_):
    tv = torchvision.models
    m = _pretrained(tv.efficientnet_b0, tv.EfficientNet_B0_Weights)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, nc)
    return m


def convnext_tiny(nc, **_):
    tv = torchvision.models
    m = _pretrained(tv.convnext_tiny, tv.ConvNeXt_Tiny_Weights)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, nc)
    return m


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # B,N,D


class ViT(nn.Module):
    def __init__(self, nc, size=224, patch=16, dim=192, depth=6, heads=3, hybrid=False):
        super().__init__()
        if hybrid:  # conv stem downsamples 8x, then 2x2 patches
            self.embed = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, dim, 2, 2), nn.Flatten(2))
            n = (size // 16) ** 2
            self.hybrid = True
        else:
            self.embed = PatchEmbed(dim, patch)
            n = (size // patch) ** 2
            self.hybrid = False
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, 0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, nc)

    def forward(self, x):
        t = self.embed(x)
        if self.hybrid:
            t = t.transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.head(self.norm(self.enc(t))[:, 0])


def vit(nc, size=224, **_):
    return ViT(nc, size)


def hybrid(nc, size=224, **_):
    return ViT(nc, size, hybrid=True)


class SelectiveSSM(nn.Module):
    """Minimal Mamba S6 block (sequential scan, pure PyTorch)."""
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        di = dim * expand
        self.in_proj = nn.Linear(dim, di * 2)
        self.conv = nn.Conv1d(di, di, 3, padding=1, groups=di)
        self.x_proj = nn.Linear(di, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out = nn.Linear(di, dim)
        self.ds = d_state

    def scan(self, x):  # x: B,L,di
        B, L, di = x.shape
        dt, Bm, Cm = self.x_proj(x).split([1, self.ds, self.ds], -1)
        dt = F.softplus(self.dt_proj(dt))                   # B,L,di
        A = -torch.exp(self.A_log)                          # di,ds
        dA = torch.exp(dt.unsqueeze(-1) * A)                # B,L,di,ds
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)
        h = x.new_zeros(B, di, self.ds)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dBx[:, i]
            ys.append((h * Cm[:, i].unsqueeze(1)).sum(-1))
        return torch.stack(ys, 1) + x * self.D

    def forward(self, u):
        x, z = self.in_proj(u).chunk(2, -1)
        x = F.silu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        y = self.scan(x)
        return self.out(y * F.silu(z))


class VimBlock(nn.Module):
    """Bidirectional: forward and flipped scans, as in Vision Mamba."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.f = SelectiveSSM(dim)
        self.b = SelectiveSSM(dim)

    def forward(self, x):
        h = self.norm(x)
        return x + self.f(h) + self.b(h.flip(1)).flip(1)


class VisionMamba(nn.Module):
    def __init__(self, nc, size=224, patch=16, dim=192, depth=6):
        super().__init__()
        self.embed = PatchEmbed(dim, patch)
        self.pos = nn.Parameter(torch.randn(1, (size // patch) ** 2, dim) * 0.02)
        self.blocks = nn.Sequential(*[VimBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, nc)

    def forward(self, x):
        t = self.blocks(self.embed(x) + self.pos)
        return self.head(self.norm(t).mean(1))


def vision_mamba(nc, size=224, **_):
    return VisionMamba(nc, size)


REGISTRY = {"resnet50": resnet50, "efficientnet_b0": efficientnet_b0,
            "convnext_tiny": convnext_tiny, "vit": vit,
            "hybrid": hybrid, "vision_mamba": vision_mamba}
