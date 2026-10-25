import torch
import torch.nn as nn


class ReliabilityGuidedFusion(nn.Module):
    """
    Spatial reliability-guided fusion for paired RGB and thermal features.

    Inputs:
        rgb_features:     [B, C, H, W]
        thermal_features: [B, C, H, W]

    Returns:
        fused_features: [B, C, H, W]
        reliability:    [B, 2, H, W]
                        channel 0 = RGB reliability
                        channel 1 = thermal reliability

    Reliability is normalized across the two modalities at each pixel.
    """

    def __init__(self, channels: int, hidden_channels: int = None):
        super().__init__()

        if channels < 1:
            raise ValueError("channels must be >= 1")

        if hidden_channels is None:
            hidden_channels = max(channels // 4, 16)

        self.channels = channels

        self.reliability_head = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1)
        )

        # Start with equal RGB and thermal reliability at every pixel.
        final_conv = self.reliability_head[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    def forward(self, rgb_features, thermal_features):
        if rgb_features.ndim != 4 or thermal_features.ndim != 4:
            raise ValueError("Both inputs must have shape [B, C, H, W]")

        if rgb_features.shape != thermal_features.shape:
            raise ValueError(
                "RGB and thermal feature shapes must match; got "
                f"{tuple(rgb_features.shape)} and "
                f"{tuple(thermal_features.shape)}"
            )

        if rgb_features.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, "
                f"received {rgb_features.shape[1]}"
            )

        joint_features = torch.cat(
            [rgb_features, thermal_features], dim=1
        )

        reliability_logits = self.reliability_head(joint_features)
        reliability = torch.softmax(reliability_logits, dim=1)

        rgb_weight = reliability[:, 0:1]
        thermal_weight = reliability[:, 1:2]

        fused_features = (
            rgb_weight * rgb_features
            + thermal_weight * thermal_features
        )

        return fused_features, reliability
