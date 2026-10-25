
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CRD_ROOT = Path("/home/cse-sdpl/Downloads/CRD_Net")
E5_ROOT = Path("/home/cse-sdpl/Downloads/CMX_E5")
CHECKPOINT_PATH = CRD_ROOT / "checkpoints" / "crd_epoch-last.pth"

sys.path.insert(0, str(E5_ROOT))

from config import config as cfg
from dataloader.dataloader import ValPre
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder
from engine.evaluator import Evaluator


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


integration = load_module(
    "crd_diagnostic_integration",
    CRD_ROOT / "models" / "crd_integration.py",
)


class DiagnosticEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        image = data["data"]
        label = data["label"]
        thermal = data["modal_x"]

        prediction = self.sliding_eval_rgbX(
            image, thermal, cfg.eval_crop_size, cfg.eval_stride_rate, device
        )

        pred = np.asarray(prediction).astype(np.int64)
        gt = np.asarray(label).astype(np.int64)

        valid_gt = (gt >= 0) & (gt < cfg.num_classes)
        valid_pred = (pred >= 0) & (pred < cfg.num_classes)

        gt_counts = np.bincount(
            gt[valid_gt], minlength=cfg.num_classes
        )
        pred_counts = np.bincount(
            pred[valid_pred], minlength=cfg.num_classes
        )

        valid_pixels = int(valid_gt.sum())
        correct_pixels = int(((pred == gt) & valid_gt).sum())
        return gt_counts, pred_counts, valid_pixels, correct_pixels

    def compute_metric(self, results):
        gt_counts = np.sum([r[0] for r in results], axis=0)
        pred_counts = np.sum([r[1] for r in results], axis=0)
        valid_pixels = sum(r[2] for r in results)
        correct_pixels = sum(r[3] for r in results)

        print("\nCRD-Net validation prediction distribution")
        print("-" * 88)
        print(
            f"{'Class ID':<10}{'Class name':<18}"
            f"{'GT pixels %':>15}{'Pred pixels %':>17}{'Pred pixel count':>20}"
        )
        print("-" * 88)

        names = list(cfg.class_names)
        for i in range(cfg.num_classes):
            name = names[i] if i < len(names) else f"class_{i}"
            gt_pct = 100.0 * gt_counts[i] / max(valid_pixels, 1)
            pred_pct = 100.0 * pred_counts[i] / max(pred_counts.sum(), 1)
            print(
                f"{i:<10}{name:<18}{gt_pct:>14.3f}%"
                f"{pred_pct:>16.3f}%{int(pred_counts[i]):>20,}"
            )

        foreground_gt = gt_counts[1:].sum()
        foreground_pred = pred_counts[1:].sum()

        print("-" * 88)
        print(f"Total valid GT pixels:       {valid_pixels:,}")
        print(f"Total predicted valid pixels: {int(pred_counts.sum()):,}")
        print(f"Overall pixel accuracy:       {100.0 * correct_pixels / max(valid_pixels, 1):.3f}%")
        print(f"Ground-truth foreground:      {100.0 * foreground_gt / max(valid_pixels, 1):.3f}%")
        print(f"Predicted foreground:         {100.0 * foreground_pred / max(pred_counts.sum(), 1):.3f}%")
        print("\nInterpretation: compare the predicted foreground percentage")
        print("against the ground-truth foreground percentage.")
        return "Prediction distribution diagnostic complete.\n"


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.backbone = "mit_b2_e5"

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

    dataset = RGBXDataset(setting, "val", ValPre())
    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=nn.BatchNorm2d)
    integration.replace_fusion_modules(model)
    model.to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Validation images:", len(dataset))
    print("Running read-only prediction distribution diagnostic...")

    evaluator = DiagnosticEvaluator(
        dataset,
        cfg.num_classes,
        cfg.norm_mean,
        cfg.norm_std,
        model,
        cfg.eval_scale_array,
        cfg.eval_flip,
        [0] if device.type == "cuda" else [],
        verbose=False,
        save_path=None,
        show_image=False,
    )
    evaluator.val_func = model

    evaluator.single_process_evalutation()


if __name__ == "__main__":
    main()
