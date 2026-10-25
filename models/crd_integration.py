
from pathlib import Path
import sys
from types import MethodType

import torch

E5_ROOT = Path("/home/cse-sdpl/Downloads/CMX_E5")
CRD_ROOT = Path("/home/cse-sdpl/Downloads/CRD_Net")

if str(E5_ROOT) not in sys.path:
    sys.path.insert(0, str(E5_ROOT))


def _crd_reliability_forward(self, x1, x2):
    """
    Preserve the original E3/E5 fusion computation and additionally
    expose its spatial reliability weights for CRD supervision.
    No modules or parameters are replaced.
    """
    batch, channels, height, width = x1.shape

    t1 = x1.flatten(2).transpose(1, 2)
    t2 = x2.flatten(2).transpose(1, 2)

    # Original pretrained CMX cross-modal interaction.
    t1, t2 = self.cross(t1, t2)

    f1 = t1.transpose(1, 2).reshape(
        batch, channels, height, width
    ).contiguous()
    f2 = t2.transpose(1, 2).reshape(
        batch, channels, height, width
    ).contiguous()

    logits = self.reliability_head(torch.cat([f1, f2], dim=1))
    weights = torch.softmax(logits, dim=1)

    # Keep reliability maps available to the CRD training objective.
    self.last_reliability = weights

    w_rgb = weights[:, 0:1]
    w_thermal = weights[:, 1:2]

    t1 = (w_rgb * f1).flatten(2).transpose(1, 2)
    t2 = (w_thermal * f2).flatten(2).transpose(1, 2)

    merged = torch.cat([t1, t2], dim=-1)

    # Original pretrained CMX channel embedding.
    return self.channel_emb(merged, height, width)


def replace_fusion_modules(model):
    """
    Compatibility function retained for existing CRD scripts.
    It does NOT replace fusion modules; it only exposes their
    existing reliability weights during forward passes.
    """
    if not hasattr(model, "backbone"):
        raise AttributeError("Expected model.backbone was not found.")

    backbone = model.backbone

    if not hasattr(backbone, "FFMs") or len(backbone.FFMs) != 4:
        raise RuntimeError("Expected exactly four original fusion stages.")

    for module in backbone.FFMs:
        if not hasattr(module, "cross"):
            raise RuntimeError("Fusion module is missing its original CrossPath.")
        if not hasattr(module, "channel_emb"):
            raise RuntimeError("Fusion module is missing its original ChannelEmbed.")
        if not hasattr(module, "reliability_head"):
            raise RuntimeError("Fusion module is missing its reliability head.")

        # Keep the original module and parameter names intact.
        module.last_reliability = None
        module.forward = MethodType(_crd_reliability_forward, module)

    backbone.crd_fusion_channels = [64, 128, 320, 512]
    return model


def load_e5_weights_into_crd(model, checkpoint_path, map_location="cpu"):
    """
    Load the complete E5 checkpoint strictly. The model architecture
    remains unchanged, including all original pretrained fusion weights.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"E5 checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("Expected a checkpoint containing the 'model' key.")

    state = checkpoint["model"]
    incompatible = model.load_state_dict(state, strict=True)

    return {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_iteration": checkpoint.get("iteration"),
        "loaded_tensor_count": len(state),
        "loaded_parameter_elements": sum(
            value.numel() for value in state.values()
            if torch.is_tensor(value)
        ),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "skipped_source_keys": [],
    }
