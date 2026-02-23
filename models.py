import torch
import torch.nn as nn
import torchvision.models as tvm


def _resnet18_backbone(in_ch: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Returns ResNet18 trunk WITHOUT avgpool/fc, output feature map: (B, 512, H', W').
    """
    m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)

    if in_ch != 3:
        # Replace first conv to accept depth=1 etc.
        old = m.conv1
        m.conv1 = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=old.bias is not None)
    trunk = nn.Sequential(
        m.conv1, m.bn1, m.relu, m.maxpool,
        m.layer1, m.layer2, m.layer3, m.layer4
    )
    return trunk


class LateFusionResNet18(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, depth_in_ch: int = 1):
        super().__init__()
        self.rgb = _resnet18_backbone(in_ch=3, pretrained=pretrained)
        self.depth = _resnet18_backbone(in_ch=depth_in_ch, pretrained=pretrained)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # rgb: (B,3,H,W), depth: (B,1,H,W)
        fr = self.rgb(rgb)          # (B,512,h,w)
        fd = self.depth(depth)      # (B,512,h,w)
        zr = self.pool(fr).flatten(1)  # (B,512)
        zd = self.pool(fd).flatten(1)  # (B,512)
        z = torch.cat([zr, zd], dim=1) # (B,1024)
        return self.head(z)


class CrossAttnFusionResNet18(nn.Module):
    """
    Token fusion: treat spatial grid features as tokens and apply cross-attention.
    """
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        depth_in_ch: int = 1,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rgb = _resnet18_backbone(in_ch=3, pretrained=pretrained)
        self.depth = _resnet18_backbone(in_ch=depth_in_ch, pretrained=pretrained)

        self.bidirectional = bidirectional

        # MHA expects (B, N, C) if batch_first=True
        self.cross_rgb = nn.MultiheadAttention(
            embed_dim=512, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm_rgb = nn.LayerNorm(512)

        if bidirectional:
            self.cross_depth = nn.MultiheadAttention(
                embed_dim=512, num_heads=num_heads, dropout=attn_dropout, batch_first=True
            )
            self.norm_depth = nn.LayerNorm(512)

        self.mlp = nn.Sequential(
            nn.Linear(512 * (2 if bidirectional else 1), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(proj_dropout),
            nn.Linear(512, num_classes)
        )

    @staticmethod
    def _to_tokens(feat: torch.Tensor) -> torch.Tensor:
        # feat: (B,C,h,w) -> tokens: (B, N=h*w, C)
        return feat.flatten(2).transpose(1, 2).contiguous()

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        fr = self.rgb(rgb)      # (B,512,h,w)
        fd = self.depth(depth)  # (B,512,h,w)

        tr = self._to_tokens(fr)  # (B,N,512)
        td = self._to_tokens(fd)  # (B,N,512)

        # RGB tokens attend to Depth tokens
        attn_r, _ = self.cross_rgb(query=tr, key=td, value=td, need_weights=False)
        tr2 = self.norm_rgb(tr + attn_r)  # residual + norm

        pr = tr2.mean(dim=1)  # (B,512)

        if not self.bidirectional:
            return self.mlp(pr)

        # Depth tokens attend to RGB tokens (optional)
        attn_d, _ = self.cross_depth(query=td, key=tr, value=tr, need_weights=False)
        td2 = self.norm_depth(td + attn_d)
        pd = td2.mean(dim=1)  # (B,512)

        z = torch.cat([pr, pd], dim=1)  # (B,1024)
        return self.mlp(z)