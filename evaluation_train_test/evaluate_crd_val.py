
import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

CRD_ROOT = Path("/home/cse-sdpl/Downloads/CRD_Net")
E5_ROOT = Path("/home/cse-sdpl/Downloads/CMX_E5")
CRD_CHECKPOINT = CRD_ROOT / "checkpoints" / "crd_best_miou.pth"

sys.path.insert(0, str(E5_ROOT))

from config import config as cfg
from dataloader.dataloader import ValPre
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder
from engine.evaluator import Evaluator
from utils.metric import hist_info, compute_score


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


integration = load_module_from_path(
    "crd_validation_integration",
    CRD_ROOT / "models" / "crd_integration.py",
)


class CRDValidationEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        image = data["data"]
        label = data["label"]
        thermal = data["modal_x"]

        prediction = self.sliding_eval_rgbX(
            image, thermal, cfg.eval_crop_size, cfg.eval_stride_rate, device
        )

        hist, labeled, correct = hist_info(cfg.num_classes, prediction, label)
        return hist, labeled, correct

    def compute_metric(self, results):
        hist = sum((item[0] for item in results))
        labeled = sum((item[1] for item in results))
        correct = sum((item[2] for item in results))

        score = compute_score(hist, correct, labeled)
        if isinstance(score, dict):
            metric_text = "\n".join(f"{key}: {value}" for key, value in score.items())
        else:
            metric_text = str(score)

        print("\nCRD-Net VALIDATION metrics")
        print(metric_text)
        return metric_text + "\n"


def main():
    if not CRD_CHECKPOINT.is_file():
        raise FileNotFoundError(f"CRD checkpoint not found: {CRD_CHECKPOINT}")

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

    # Verify CRD checkpoint structure before evaluation.
    checkpoint = torch.load(CRD_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")

    print("CRD-Net TEST setup")
    print("  Device:", device)
    print("  Checkpoint:", CRD_CHECKPOINT)
    print("  Checkpoint epoch:", checkpoint.get("epoch"))
    print("  Validation images:", len(dataset))
    print("  Evaluation scales:", cfg.eval_scale_array)
    print("  Flip inference:", cfg.eval_flip)
    print("  Crop size:", cfg.eval_crop_size)
    print("  Sliding stride rate:", cfg.eval_stride_rate)

    evaluator = CRDValidationEvaluator(
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

    log_path = CRD_ROOT / "logs" / "crd_val_best.log"
    link_path = CRD_ROOT / "logs" / "crd_test_best_last.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator.run(
        str(CRD_CHECKPOINT.parent),
        str(CRD_CHECKPOINT),
        str(log_path),
        str(link_path),
    )

    print("\nTest evaluation complete.")
    print("Test log:", log_path)


if __name__ == "__main__":
    main()
