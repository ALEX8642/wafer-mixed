"""
train.py — Multi-label training loop: BCE-with-logits, AdamW, cosine LR,
early stopping on val macro-F1 over the 8 labels.

Ported from wafer-defect-classifier's train.py with the multi-class pieces
swapped out:
    - CrossEntropy/Focal → BCEWithLogitsLoss. No pos_weight, despite real
      per-label skew (Near-full 149/38,015 positives ≈ 1:255, Random 866):
      a 1-epoch smoke run already reaches recall 1.0 / 0.92 on those two at
      the 0.5 threshold, so unweighted BCE is the honest baseline. Revisit
      with pos_weight or the Phase 3 per-label thresholds only if the
      converged run shows rare-label collapse.
    - argmax predictions → sigmoid at metrics.DEFAULT_THRESHOLD multi-hot.
    - class_weights / class_map.json → fixed LABEL_NAMES saved in the ckpt.

The backbone_ckpt_path hook is ported as-is; Phase 1 trains from scratch
(backbone_ckpt_path stays empty), Phase 2 points it at the WM-811K and
wafer-ssl backbones for the transfer study.

Entry point: python -m wafer_mixed.train [--config configs/baseline.yaml] [overrides...]
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from wafer_mixed.config import MixedConfig, build_arg_parser
from wafer_mixed.data import LABEL_NAMES, get_dataloaders
from wafer_mixed.metrics import macro_f1, predict_multihot
from wafer_mixed.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device_type: str,
) -> tuple[float, float]:
    """Run one train or eval epoch. Returns (avg_loss, macro_f1@0.5)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    ctx = torch.enable_grad if training else torch.no_grad
    with ctx():
        for inputs, targets_cpu in tqdm(loader, leave=False, desc="train" if training else "val"):
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets_cpu.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                logits = model(inputs)
                loss   = criterion(logits, targets)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * inputs.size(0)
            # all_preds.append(predict_multihot(torch.sigmoid(logits.float()).cpu().numpy()))
            all_preds.append(predict_multihot(torch.sigmoid(logits.float()).detach().cpu().numpy()))
	    # the loader already gave us targets on CPU — no round trip needed
            all_targets.append(targets_cpu.numpy().astype(np.int64))

    avg_loss = total_loss / len(loader.dataset)
    f1 = macro_f1(np.concatenate(all_targets), np.concatenate(all_preds))
    return avg_loss, f1


def train(cfg: MixedConfig) -> None:
    set_seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    print(f"Device: {device}  |  arch: {cfg.arch}  |  cbam: {cfg.cbam}  |  "
          f"pretrained: {cfg.pretrained}")
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(cfg)
    print(f"  Labels: {LABEL_NAMES}")

    model = build_model(cfg).to(device)
    if cfg.backbone_ckpt_path:
        bb_ckpt = torch.load(cfg.backbone_ckpt_path, map_location=device, weights_only=False)
        bb_state = bb_ckpt["backbone_state_dict"]
        # Drop any classifier head from the source: a 9-class WM-811K head
        # (9,512) against our 8-logit fc would make load_state_dict raise on
        # the shape mismatch even with strict=False. Heads never transfer.
        bb_state = {k: v for k, v in bb_state.items() if not k.startswith("fc.")}
        missing, unexpected = model.load_state_dict(bb_state, strict=False)
        # Actually-loaded = source keys the model accepted = source - unexpected.
        n_loaded = len(bb_state) - len(unexpected)
        print(f"Pretrained backbone: loaded {n_loaded}/{len(bb_state)} tensors from "
              f"{cfg.backbone_ckpt_path}  (missing={len(missing)}, unexpected={len(unexpected)})")
        # Guard: an architecture mismatch (e.g. cbam flag differs) silently
        # loads almost nothing and wastes the pretraining. Fail loud instead.
        if n_loaded < 0.5 * len(bb_state):
            raise RuntimeError(
                f"Only {n_loaded}/{len(bb_state)} pretrained tensors matched the model. "
                f"Architecture mismatch (likely cbam flag differs between pretrain and "
                f"fine-tune). Pretrain with the SAME cbam setting as baseline.yaml."
            )

    criterion = nn.BCEWithLogitsLoss()
    print("Loss: BCEWithLogitsLoss (8 independent labels, no pos_weight — "
          "rare labels verified learnable without it; see module docstring)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    scaler    = torch.amp.GradScaler(enabled=(device_type == "cuda"))

    # Start below any reachable F1 so epoch 1 always writes a checkpoint:
    # otherwise a diverged run could early-stop with no best.pt, and a later
    # evaluate would silently pick up a stale checkpoint from a previous run.
    best_val_f1    = -1.0
    patience_count = 0
    ckpt_path      = cfg.output_dir / "best.pt"

    for epoch in range(1, cfg.num_epochs + 1):
        tr_loss, tr_f1 = _epoch(
            model, train_loader, criterion, device, optimizer, scaler, device_type
        )
        va_loss, va_f1 = _epoch(
            model, val_loader, criterion, device, None, scaler, device_type
        )
        scheduler.step()

        improved = va_f1 > best_val_f1
        marker   = " *" if improved else ""
        print(
            f"Epoch {epoch:3d}/{cfg.num_epochs}  "
            f"train loss {tr_loss:.4f} f1 {tr_f1:.4f}  |  "
            f"val loss {va_loss:.4f} f1 {va_f1:.4f}{marker}"
        )

        if improved:
            best_val_f1    = va_f1
            patience_count = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1":     best_val_f1,
                    "label_names":      LABEL_NAMES,
                    "cfg":              cfg.to_dict(),
                },
                ckpt_path,
            )
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"Early stop: no val macro-F1 gain for {cfg.patience} epochs.")
                break

    print(f"\nDone. Best val macro-F1 (8 labels @0.5): {best_val_f1:.4f}")
    print(f"Checkpoint : {ckpt_path}")


if __name__ == "__main__":
    parser = build_arg_parser("wafer-mixed train")
    args   = parser.parse_args()
    cfg    = MixedConfig.from_yaml_and_args(args.config, args)
    train(cfg)
