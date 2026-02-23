import torch
import torch.nn.functional as F


def depth_dropout(depth: torch.Tensor, mode: str = "zero") -> torch.Tensor:
    """
    depth: (B,1,H,W)
    """
    if mode == "zero":
        return torch.zeros_like(depth)
    if mode == "noise":
        return torch.rand_like(depth)
    raise ValueError("mode must be 'zero' or 'noise'")


def rgb_occlusion(rgb: torch.Tensor, frac: float = 0.25) -> torch.Tensor:
    """
    Simple square occlusion patch. rgb: (B,3,H,W)
    """
    b, c, h, w = rgb.shape
    patch = int((h * w * frac) ** 0.5)
    patch = max(8, min(patch, min(h, w) // 2))

    out = rgb.clone()
    y0 = (h - patch) // 2
    x0 = (w - patch) // 2
    out[:, :, y0:y0 + patch, x0:x0 + patch] = 0.0
    return out


def rgb_dark(rgb: torch.Tensor, factor: float = 0.3) -> torch.Tensor:
    return torch.clamp(rgb * factor, 0.0, 1.0)


def rgb_blur(rgb: torch.Tensor, k: int = 7) -> torch.Tensor:
    # Cheap blur via avg pooling (not a real gaussian, but fine for stress test)
    pad = k // 2
    x = F.pad(rgb, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=k, stride=1)