import math

import torch
from torch import nn
from torchvision.models import resnet18


class LinearPatchEmbed(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvStemPatchEmbed(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        if patch_size < 2 or patch_size & (patch_size - 1) != 0:
            raise ValueError("patch_size must be a power of two")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        layers = []
        channels = in_channels
        hidden = max(32, embed_dim // 2)
        self.patch_size = patch_size
        num_downsamples = int(math.log2(patch_size))
        for index in range(num_downsamples):
            out_channels = hidden if index < num_downsamples - 1 else embed_dim
            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
            channels = out_channels

        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.0,
        patch_embed="linear",
    ):
        super().__init__()
        if patch_embed == "linear":
            self.patch_embed = LinearPatchEmbed(
                image_size=image_size, patch_size=patch_size, embed_dim=embed_dim
            )
        elif patch_embed == "convstem":
            self.patch_embed = ConvStemPatchEmbed(
                image_size=image_size, patch_size=patch_size, embed_dim=embed_dim
            )
        else:
            raise ValueError(f"Unknown patch embedding: {patch_embed}")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def build_resnet18(num_classes=10, image_size=32):
    model = resnet18(weights=None, num_classes=num_classes)
    if image_size < 64:
        # 小图适配（CIFAR 风格）：3x3 conv, 无 maxpool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def create_model(config):
    model_cfg = config["model"]
    name = model_cfg["name"].lower()
    num_classes = int(config["data"].get("num_classes", 10))

    if name == "resnet18":
        return build_resnet18(
            num_classes=num_classes,
            image_size=int(config["data"].get("image_size", 32)),
        )

    if name not in {"vanilla_vit", "convstem_vit"}:
        raise ValueError(f"Unsupported model: {name}")

    patch_embed = "linear" if name == "vanilla_vit" else "convstem"
    return VisionTransformer(
        image_size=int(config["data"].get("image_size", 32)),
        patch_size=int(model_cfg.get("patch_size", 4)),
        num_classes=num_classes,
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        depth=int(model_cfg.get("depth", 6)),
        num_heads=int(model_cfg.get("num_heads", 3)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        patch_embed=patch_embed,
    )
