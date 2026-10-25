
import os
import sys
import argparse
import random
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CRD_ROOT = Path("/home/cse-sdpl/Downloads/CRD_Net")
E5_ROOT = Path("/home/cse-sdpl/Downloads/CMX_E5")
E5_CHECKPOINT = (
    E5_ROOT / "log_TDD_mit_b2_e5" / "checkpoint" / "epoch-100.pth"
)

# Keep the original CMX imports available, without editing E5 files.
sys.path.insert(0, str(E5_ROOT))

from config import config as cfg
from dataloader.dataloader import TrainPre
from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import ValPre
from engine.evaluator import Evaluator
from utils.metric import hist_info, compute_score
from models.builder import EncoderDecoder


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


integration = load_module_from_path(
    "crd_integration_module",
    CRD_ROOT / "models" / "crd_integration.py",
)
counterfactual = load_module_from_path(
    "crd_counterfactual_module",
    CRD_ROOT / "models" / "counterfactual_reliability.py",
)

replace_fusion_modules = integration.replace_fusion_modules
load_e5_weights_into_crd = integration.load_e5_weights_into_crd
build_counterfactual_targets = counterfactual.build_counterfactual_targets
counterfactual_reliability_loss = counterfactual.counterfactual_reliability_loss


def make_data_loader(batch_size, steps, workers):
    setting = {
        "rgb_root": cfg.rgb_root_folder,
        "rgb_format": cfg.rgb_format,
        "gt_root": cfg.gt_root_folder,
        "gt_format": cfg.gt_format,
        "transform_gt": cfg.gt_transform,
        "x_root": cfg.x_root_folder,
        "x_format": cfg.x_format,
        "x_single_channel": cfg.x_is_single_channel,
        "class_names": cfg.class_names,
        "train_source": cfg.train_source,
        "eval_source": cfg.eval_source,
        "test_source": cfg.test_source,
    }
    preprocess = TrainPre(cfg.norm_mean, cfg.norm_std)
    dataset = RGBXDataset(
        setting,
        "train",
        preprocess,
        file_length=batch_size * steps,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def corrupt_modality(images, fraction=0.25):
    """Mask one random rectangle per image; inputs are already normalized."""
    corrupted = images.clone()
    batch, _, height, width = corrupted.shape
    ratio = float(fraction) ** 0.5
    rect_h = max(1, min(height, int(height * ratio)))
    rect_w = max(1, min(width, int(width * ratio)))

    for b in range(batch):
        top = random.randint(0, max(0, height - rect_h))
        left = random.randint(0, max(0, width - rect_w))
        corrupted[b, :, top:top + rect_h, left:left + rect_w] = 0

    return corrupted


def compute_batch_losses(
    model,
    teacher,
    rgb,
    thermal,
    labels,
    temperature=0.5,
    distill_temperature=2.0,
    distill_weight=0.1,
):
    # Student clean forward: segmentation gradients and fusion reliability maps.
    model.train()
    clean_logits = model.encode_decode(rgb, thermal)
    reliability_maps = [
        module.last_reliability for module in model.backbone.FFMs
    ]

    if any(item is None for item in reliability_maps):
        raise RuntimeError("A CRD fusion stage did not produce reliability maps.")

    seg_loss = F.cross_entropy(
        clean_logits,
        labels.long(),
        ignore_index=cfg.background,
    )

    # Counterfactual targets and teacher predictions do not require gradients.
    was_training = model.training
    model.eval()
    teacher.eval()

    with torch.no_grad():
        clean_cf_logits = model.encode_decode(rgb, thermal)
        rgb_cf_logits = model.encode_decode(
            corrupt_modality(rgb), thermal
        )
        thermal_cf_logits = model.encode_decode(
            rgb, corrupt_modality(thermal)
        )
        teacher_logits = teacher.encode_decode(rgb, thermal)

    model.train(was_training)

    targets, confidence, valid = build_counterfactual_targets(
        clean_cf_logits,
        rgb_cf_logits,
        thermal_cf_logits,
        labels,
        temperature=temperature,
        ignore_index=cfg.background,
    )

    rel_loss = counterfactual_reliability_loss(
        targets,
        confidence,
        valid,
        reliability_maps,
        ignore_index=cfg.background,
    )

    if clean_logits.shape != teacher_logits.shape:
        raise RuntimeError(
            "Teacher/student logit shape mismatch: "
            f"{tuple(teacher_logits.shape)} vs {tuple(clean_logits.shape)}"
        )

    # Temperature-scaled KL divergence from the frozen E5 teacher.
    distill_temperature = float(distill_temperature)
    kl_map = F.kl_div(
        F.log_softmax(clean_logits / distill_temperature, dim=1),
        F.softmax(teacher_logits / distill_temperature, dim=1),
        reduction="none",
    ).sum(dim=1)

    valid_pixels = labels != cfg.background
    if valid_pixels.any():
        distill_loss = kl_map[valid_pixels].mean()
    else:
        distill_loss = kl_map.mean()

    distill_loss = distill_loss * distill_temperature ** 2

    total_loss = (
        seg_loss
        + 0.1 * rel_loss
        + float(distill_weight) * distill_loss
    )

    return (
        total_loss,
        seg_loss.detach(),
        rel_loss.detach(),
        distill_loss.detach(),
    )


def main():
    parser = argparse.ArgumentParser(description="Isolated CRD-Net training")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Perform optimizer updates and save CRD checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=784)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest CRD checkpoint.")
    args = parser.parse_args()

    if args.epochs < 1 or args.steps_per_epoch < 1:
        raise ValueError("epochs and steps-per-epoch must be positive.")
    if not E5_CHECKPOINT.is_file():
        raise FileNotFoundError(f"E5 initialization checkpoint missing: {E5_CHECKPOINT}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep all CRD outputs separate from E5.
    for folder in ("checkpoints", "logs"):
        (CRD_ROOT / folder).mkdir(parents=True, exist_ok=True)

    cfg.backbone = "mit_b2_e5"
    cfg.batch_size = args.batch_size
    criterion = None
    model = EncoderDecoder(cfg=cfg, criterion=criterion, norm_layer=nn.BatchNorm2d)
    replace_fusion_modules(model)
    load_info = load_e5_weights_into_crd(model, str(E5_CHECKPOINT))
    model.to(device)

    print("CRD-Net initialization")
    print("  Device:", device)
    print("  E5 checkpoint:", E5_CHECKPOINT)
    print("  Checkpoint epoch:", load_info["checkpoint_epoch"])
    print("  Loaded tensors:", load_info["loaded_tensor_count"])
    print("  Missing keys (new CRD fusion parameters):", len(load_info["missing_keys"]))
    print("  Unexpected keys:", len(load_info["unexpected_keys"]))

    if load_info["unexpected_keys"]:
        raise RuntimeError(f"Unexpected checkpoint keys: {load_info['unexpected_keys']}")

    loader = make_data_loader(args.batch_size, args.steps_per_epoch, args.workers)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    start_epoch = 1
    if args.resume:
        resume_path = CRD_ROOT / "checkpoints" / "crd_epoch-last.pth"
        if not resume_path.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint missing: {resume_path}")
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        if "scheduler" in saved:
            scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved.get("epoch", 0)) + 1
        print(f"Resuming from epoch {start_epoch}; checkpoint: {resume_path}")

    if not args.run:
        print("\nSAFE DIAGNOSTIC MODE: no optimizer step and no checkpoint write.")
        model.train()
        batch = next(iter(loader))
        rgb = batch["data"].to(device, dtype=torch.float32, non_blocking=True)
        labels = batch["label"].to(device, dtype=torch.long, non_blocking=True)
        thermal = batch["modal_x"].to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        total_loss, seg_loss, rel_loss = compute_batch_losses(
            model, rgb, thermal, labels, temperature=args.temperature
        )
        if not torch.isfinite(total_loss):
            raise RuntimeError("Diagnostic total loss is not finite.")
        total_loss.backward()

        grad_stages = [
            any(
                p.grad is not None and torch.isfinite(p.grad).all()
                and p.grad.abs().sum().item() > 0
                for p in module.parameters()
            )
            for module in model.backbone.FFMs
        ]
        print("  Batch RGB shape:", tuple(rgb.shape))
        print("  Batch label shape:", tuple(labels.shape))
        print("  Segmentation loss:", float(seg_loss))
        print("  Reliability loss:", float(rel_loss))
        print("  Total loss:", float(total_loss.detach()))
        print("  Fusion stages with nonzero gradients:", grad_stages)
        if not all(grad_stages):
            raise RuntimeError("Gradient check failed for one or more CRD fusion stages.")
        print("DIAGNOSTIC PASS")
        return

    print("\nTRAINING MODE ENABLED: CRD outputs only.")
    log_path = CRD_ROOT / "logs" / "train_crd.log"
    checkpoint_path = CRD_ROOT / "checkpoints" / "crd_epoch-last.pth"

    with log_path.open("a", buffering=1) as log:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_total = 0.0
            running_seg = 0.0
            running_rel = 0.0
            completed = 0

            for step, batch in enumerate(loader):
                if step >= args.steps_per_epoch:
                    break

                rgb = batch["data"].to(device, dtype=torch.float32, non_blocking=True)
                labels = batch["label"].to(device, dtype=torch.long, non_blocking=True)
                thermal = batch["modal_x"].to(device, dtype=torch.float32, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                total_loss, seg_loss, rel_loss = compute_batch_losses(
                    model, rgb, thermal, labels, temperature=args.temperature
                )
                if not torch.isfinite(total_loss):
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch}, step {step + 1}"
                    )
                total_loss.backward()
                optimizer.step()

                running_total += float(total_loss.detach())
                running_seg += float(seg_loss)
                running_rel += float(rel_loss)
                completed += 1

            if completed == 0:
                raise RuntimeError("No training batches were completed.")

            metrics = {
                "epoch": epoch,
                "steps": completed,
                "total_loss": running_total / completed,
                "seg_loss": running_seg / completed,
                "reliability_loss": running_rel / completed,
            }
            current_lr = optimizer.param_groups[0]["lr"]
            message = (
                f"Epoch {epoch}/{args.epochs} | lr={current_lr:.2e} | steps={completed} | "
                f"total={metrics['total_loss']:.6f} | "
                f"seg={metrics['seg_loss']:.6f} | "
                f"reliability={metrics['reliability_loss']:.6f}"
            )
            print(message)
            log.write(message + "\n")
            scheduler.step()

            # One rolling checkpoint to limit disk consumption.
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "metrics": metrics,
                    "source_checkpoint": str(E5_CHECKPOINT),
                },
                checkpoint_path,
            )

    print("Training finished. Rolling CRD checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
