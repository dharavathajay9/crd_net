
import torch
import torch.nn.functional as F


def build_counterfactual_targets(
    clean_logits,
    rgb_corrupted_logits,
    thermal_corrupted_logits,
    labels,
    temperature=0.5,
    ignore_index=255,
):
    """
    Build detached RGB/thermal reliability targets from counterfactual
    segmentation losses, following the E5 target-generation approach.
    Logits have shape [B, classes, H, W]; labels have shape [B, H, W].
    """
    if clean_logits.shape != rgb_corrupted_logits.shape:
        raise ValueError("Clean and RGB-corrupted logits must match.")
    if clean_logits.shape != thermal_corrupted_logits.shape:
        raise ValueError("Clean and thermal-corrupted logits must match.")
    if labels.shape != clean_logits.shape[:1] + clean_logits.shape[2:]:
        raise ValueError("Labels must have shape [B, H, W].")

    labels = labels.long()
    valid = labels != ignore_index

    with torch.no_grad():
        clean_loss = F.cross_entropy(
            clean_logits, labels, ignore_index=ignore_index, reduction="none"
        )
        rgb_loss = F.cross_entropy(
            rgb_corrupted_logits, labels,
            ignore_index=ignore_index, reduction="none"
        )
        thermal_loss = F.cross_entropy(
            thermal_corrupted_logits, labels,
            ignore_index=ignore_index, reduction="none"
        )

        delta_rgb = (rgb_loss - clean_loss).clamp_min(0)
        delta_thermal = (thermal_loss - clean_loss).clamp_min(0)

        def robust_normalize(delta):
            normalized = torch.zeros_like(delta)

            for b in range(delta.shape[0]):
                values = delta[b][valid[b]].float()
                if values.numel() == 0:
                    continue

                q25, median, q75 = torch.quantile(
                    values,
                    torch.tensor(
                        [0.25, 0.50, 0.75],
                        device=values.device,
                        dtype=values.dtype,
                    ),
                ).unbind()

                scale = q75 - q25
                if scale.item() < 1e-6:
                    positive = values[values > 1e-8]
                    scale = (
                        torch.quantile(positive, 0.90).clamp_min(1e-6)
                        if positive.numel()
                        else values.new_tensor(1.0)
                    )

                normalized[b] = (
                    (delta[b] - median) / scale
                ).clamp(-3.0, 3.0)

            return normalized

        norm_rgb = robust_normalize(delta_rgb)
        norm_thermal = robust_normalize(delta_thermal)

        evidence_difference = (delta_rgb - delta_thermal).abs()
        confidence = torch.sigmoid((evidence_difference - 0.02) / 0.01)
        confidence = confidence * valid.float()

        normalized_deltas = torch.stack([norm_rgb, norm_thermal], dim=1)
        informative_target = torch.softmax(
            normalized_deltas / max(float(temperature), 1e-6), dim=1
        )

        uniform_prior = torch.full_like(informative_target, 0.5)
        targets = (
            (1.0 - confidence.unsqueeze(1)) * uniform_prior
            + confidence.unsqueeze(1) * informative_target
        ).detach()

    return targets, confidence.detach(), valid


def counterfactual_reliability_loss(
    targets, confidence, valid, reliability_maps, ignore_index=255
):
    """Confidence-weighted reliability cross-entropy across feature stages."""
    if not reliability_maps or len(reliability_maps) != 4:
        raise ValueError("Expected four reliability maps.")

    stage_losses = []

    for reliability in reliability_maps:
        if reliability.ndim != 4 or reliability.shape[1] != 2:
            raise ValueError("Each reliability map must have shape [B, 2, H, W].")

        target = F.interpolate(
            targets, size=reliability.shape[-2:],
            mode="bilinear", align_corners=False
        )
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)

        confidence_stage = F.interpolate(
            confidence.unsqueeze(1), size=reliability.shape[-2:],
            mode="bilinear", align_corners=False
        ).squeeze(1)

        valid_stage = F.interpolate(
            valid.unsqueeze(1).float(), size=reliability.shape[-2:],
            mode="nearest"
        ).squeeze(1)

        pixel_loss = -(
            target * reliability.clamp_min(1e-6).log()
        ).sum(dim=1)

        weights = confidence_stage * valid_stage
        stage_loss = (
            (pixel_loss * weights).sum() / weights.sum().clamp_min(1e-6)
        )
        stage_losses.append(stage_loss)

    return torch.stack(stage_losses).mean()
