"""
explain.py — Grad-CAM / Grad-CAM++ for multi-label mixed-defect predictions.

Ported from wafer-defect-classifier's explain.py with the multi-class pieces
swapped for multi-label:
    - softmax probs → per-label sigmoid.
    - "predicted class" → an explicit target *label*: with several defects on
      one map there is no single argmax; the caller names the label whose
      evidence to localise, and the default is the highest-sigmoid label.
    - example generator walks *mixed* maps and renders one CAM per active
      label on the same map — the Phase 3 question is whether attention
      separates the superposed signatures (Scratch heat distinct from
      Edge-Ring heat on the same wafer).

Target layer: the last BasicBlock of layer4. With cbam=true our model wraps
each stage as Sequential(original_stage, CBAM) — nested, unlike the main
repo's flat Sequential — so the lookup unwraps the inner Sequential and
skips the CBAM module (its output is attention-rescaled, which distorts CAM
weighting).

Entry point: python -m wafer_mixed.explain [--checkpoint outputs/best.pt]
Writes per-combo overlay figures to outputs/grad_cam/. Phase 3 curates a
subset into assets/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wafer_mixed.config import MixedConfig, build_arg_parser
from wafer_mixed.data import LABEL_NAMES, combo_name, get_dataloaders
from wafer_mixed.metrics import DEFAULT_THRESHOLD
from wafer_mixed.model import CBAM, load_checkpoint_model


# ---------------------------------------------------------------------------
# GradCAM engines (hook-based, ported verbatim apart from sigmoid targets)
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Context-manager GradCAM for one label of a multi-label CNN.

    Usage:
        with GradCAM(model, target_layer(model)) as cam:
            heatmap, label_idx, probs = cam.compute(input_tensor, target_label=6)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []

    def __enter__(self) -> "GradCAM":
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_activations)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_gradients)
        )
        return self

    def __exit__(self, *_) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _save_activations(self, _module, _input, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def _forward(
        self, input_tensor: torch.Tensor, target_label: Optional[int]
    ) -> tuple[torch.Tensor, int, np.ndarray]:
        """Shared forward + backward-on-target-logit. Returns (x, label, probs)."""
        self.model.eval()
        x = input_tensor.clone().requires_grad_(True)

        logits = self.model(x)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()

        if target_label is None:
            target_label = int(np.argmax(probs))

        self.model.zero_grad()
        logits[0, target_label].backward()
        return x, target_label, probs

    def _to_heatmap(self, cam: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return F.interpolate(
            cam, size=size, mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_label: Optional[int] = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Run GradCAM for one sample and one label.

        Args:
            input_tensor: (1, C, H, W) on the same device as model.
            target_label: label index to explain; None = highest-sigmoid label.

        Returns:
            heatmap (H, W) in [0, 1], explained label index, sigmoid probs (8,)
        """
        _, target_label, probs = self._forward(input_tensor, target_label)

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, Ch, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        heatmap = self._to_heatmap(cam, input_tensor.shape[-2:])
        return heatmap, target_label, probs


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ (Chattopadhyay et al., 2018) — sharper localisation when the
    defect occupies a small fraction of the wafer (Scratch streaks, Loc
    clusters), which matters double here: inside a mix each label's evidence
    is by construction only part of the map. Weight formula and rationale as
    in the main repo's explain.py.
    """

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_label: Optional[int] = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        _, target_label, probs = self._forward(input_tensor, target_label)

        A = self._activations         # (1, Ch, h, w)
        G = self._gradients           # (1, Ch, h, w)

        G2 = G ** 2
        G3 = G ** 3
        denom = 2.0 * G2 + (A * G3).sum(dim=(2, 3), keepdim=True)
        alpha = G2 / (denom + 1e-8)                                # (1, Ch, h, w)
        weights = (alpha * F.relu(G)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * A).sum(dim=1, keepdim=True)
        heatmap = self._to_heatmap(cam, input_tensor.shape[-2:])
        return heatmap, target_label, probs


def target_layer(model: nn.Module) -> nn.Module:
    """
    Last BasicBlock of layer4 — the deepest spatial features before pooling.
    cbam=true wraps the stage as Sequential(original_layer4, CBAM); unwrap
    the inner Sequential and never target the CBAM module itself.
    """
    for m in reversed(list(model.layer4.children())):
        if isinstance(m, CBAM):
            continue
        return m[-1] if isinstance(m, nn.Sequential) else m
    raise ValueError("layer4 contains no non-CBAM block to target")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """
    (3, H, W) one-hot tensor → (H, W) uint8 image.
    Channel order [outside=0, pass=1, fail=2] → pixel values 40/160/255.
    """
    wafer_map = tensor.argmax(dim=0).cpu().numpy().astype(np.uint8)
    lut = np.array([40, 160, 255], dtype=np.uint8)
    return lut[wafer_map]


def save_multilabel_overlay(
    wafer_img: np.ndarray,
    heatmaps: dict[int, np.ndarray],
    probs: np.ndarray,
    y_true: np.ndarray,
    save_path: Path,
) -> None:
    """
    One row: wafer map, then one CAM overlay per explained label — the
    superposition-separation figure. `heatmaps` maps label index → (H, W) CAM.
    Jet blend kept identical to the source repos' Grad-CAM assets for visual
    continuity across the portfolio.
    """
    n = 1 + len(heatmaps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.4))
    axes = np.atleast_1d(axes)

    axes[0].imshow(wafer_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Wafer map\ntrue: {combo_name(y_true)}", fontsize=11)
    axes[0].axis("off")

    wafer_rgb = np.stack([wafer_img] * 3, axis=-1).astype(float) / 255.0
    for ax, (label_idx, heatmap) in zip(axes[1:], sorted(heatmaps.items())):
        overlay = plt.cm.jet(heatmap)[..., :3]
        ax.imshow(0.55 * wafer_rgb + 0.45 * overlay)
        ax.set_title(
            f"{LABEL_NAMES[label_idx]}\nsigmoid={probs[label_idx]:.2f}",
            fontsize=11,
        )
        ax.axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point: per-label CAMs on exactly-classified mixed maps
# ---------------------------------------------------------------------------

# Mixes chosen to pair distinct spatial signatures; the 3-mix is the combo
# with doubled representation (2,000 maps) noted in Phase 0.
_EXAMPLE_COMBOS = [
    "Edge-Ring+Scratch",
    "Center+Scratch",
    "Donut+Edge-Loc",
    "Edge-Loc+Loc",
    "Center+Edge-Loc+Scratch",
]


def generate_mixed_cam_examples(
    cfg: MixedConfig,
    checkpoint_path: Path | None = None,
    method: str = "gradcampp",
    combos: list[str] | None = None,
) -> list[Path]:
    """
    For each requested combo, find one exactly-classified test map and save a
    figure with one CAM per true label. Returns the written paths.
    """
    if checkpoint_path is None:
        checkpoint_path = cfg.output_dir / "best.pt"
    combos = _EXAMPLE_COMBOS if combos is None else combos

    model, _ = load_checkpoint_model(cfg, checkpoint_path)

    cam_cls = GradCAMPlusPlus if method == "gradcampp" else GradCAM
    print(f"  Method: {cam_cls.__name__}")

    _, _, test_loader = get_dataloaders(cfg)

    wanted = {c: None for c in combos}
    with torch.no_grad():
        for inputs, targets in test_loader:
            logits = model(inputs.to(cfg.device))
            preds = (torch.sigmoid(logits) > DEFAULT_THRESHOLD).cpu().numpy()
            trues = targets.numpy().astype(np.int64)
            for i in range(len(trues)):
                name = combo_name(trues[i])
                if wanted.get(name) is None and name in wanted \
                        and (preds[i] == trues[i]).all():
                    wanted[name] = (inputs[i], trues[i])
            if all(v is not None for v in wanted.values()):
                break

    out_dir = cfg.output_dir / "grad_cam"
    written: list[Path] = []
    with cam_cls(model, target_layer(model)) as cam:
        for name, found in wanted.items():
            if found is None:
                print(f"  Warning: no exactly-classified test map for {name}")
                continue
            tensor, y_true = found
            inp = tensor.unsqueeze(0).to(cfg.device)
            heatmaps: dict[int, np.ndarray] = {}
            probs = None
            for label_idx in np.flatnonzero(y_true):
                heatmap, _, probs = cam.compute(inp, target_label=int(label_idx))
                heatmaps[int(label_idx)] = heatmap
            path = out_dir / f"gradcam_{name.lower().replace('+', '_').replace('-', '_')}.png"
            save_multilabel_overlay(
                tensor_to_display(tensor), heatmaps, probs, y_true, path
            )
            written.append(path)
            print(f"  Saved: {path}")

    print(f"\nMixed-pattern CAM figures in {out_dir}/")
    return written


if __name__ == "__main__":
    parser = build_arg_parser("wafer-mixed explain")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--method", choices=["gradcam", "gradcampp"], default="gradcampp",
        help="gradcampp (default, sharper on partial-map evidence) or gradcam",
    )
    args = parser.parse_args()
    cfg = MixedConfig.from_yaml_and_args(args.config, args)
    print(f"Generating {args.method} examples for: {_EXAMPLE_COMBOS}")
    generate_mixed_cam_examples(cfg, checkpoint_path=args.checkpoint, method=args.method)
