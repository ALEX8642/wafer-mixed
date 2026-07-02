# STATUS — wafer-mixed

Session handoff log. One phase per session (see workspace `PLAN-wafer-mixed.md`).

## Phase 0 — Scaffold + data ✅ (2026-07-01)

**Done:**
- Repo scaffold mirroring `wafer-defect-classifier` (src/wafer_mixed, configs,
  tests, scripts, docs, assets). MIT license, data gitignored.
- `scripts/download_data.py`: auto-downloads MixedWM38 (~412 MB) from the
  authors' Google Drive, SHA256-verifies, checks shapes/combos, writes the
  persisted split.
- `src/wafer_mixed/data.py`: loader yields (3×S×S one-hot tensor, 8-dim
  float multi-hot). Split stratified by full 38-type combination, seed 42,
  persisted to `data/splits.npz` (committed): train 26,610 / val 3,802 /
  test 7,603.
- `scripts/eda.py` → `docs/DATA.md` (frequency tables, sample grids) +
  4 figure grids in `assets/`.
- Tests: 8 passing — encode round-trip, clip, multi-hot correctness, split
  leakage (disjoint + full coverage), stratification (38 combos in every
  split), loader-vs-raw label agreement.

**Dataset facts verified (deviations from the plan's assumptions):**
- 38,015 maps, 52×52, 38 combos, 8-dim multi-hot — all as assumed.
- **Label ordering was undocumented upstream**; verified visually:
  `[Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Scratch, Random]`
  (see docs/DATA.md + assets/singles_grid.png).
- **Pixel values include a stray 3** (214 px / 105 maps) — clipped to 2 in
  `encode_map`, documented in DATA.md.
- **Not GAN-uniform everywhere:** Near-full single has only 149 maps and
  Random single 866; **neither appears in any mix**. Per-label metrics for
  those two will ride on small counts — flag this in Phase 1 analysis.
- One combo (Center+Edge-Loc+Scratch) has 2,000 maps, not 1,000.

**Next (Phase 1, fresh session):** multi-label baseline — port
ResNet-18+CBAM, 8-logit head, BCE-with-logits, D4 augmentation, metrics
module (per-label F1, macro-F1, exact-match, single-vs-mixed breakdown),
train from scratch on 5090.

## Phase 1 — Multi-label baseline 🔨 code done, 5090 run pending (2026-07-01)

**Done (implemented + tested on the 4090 laptop):**
- `model.py`: ResNet-18+CBAM ported verbatim from the main repo; head → 8
  logits (no sigmoid — BCEWithLogitsLoss). `cbam: true` in baseline.yaml.
- `train.py`: BCE-with-logits (no pos_weight — combos near-uniform), AdamW,
  cosine LR, AMP, early stop on val macro-F1@0.5. `backbone_ckpt_path` hook
  ported for Phase 2 (empty = from scratch, the Phase 1 arm).
- `data.py`: D4 augmentation on the train split only (labels are
  D4-invariant, so multi-hot targets unchanged).
- `metrics.py`: per-label F1, macro-F1 (8 labels), exact-match ratio,
  normal/single/mixed subset breakdown, per-label recall by subset, and a
  **spurious-activation matrix** S[i,j] = P(predict j | i true, j absent) —
  the multi-label analogue of a confusion matrix.
  Note: subset macro-F1 averages only labels with support in the subset
  (Near-full/Random never mix; their zero-support F1=0 would distort it).
- `evaluate.py`: full report + per_label_metrics.csv, metrics.json,
  spurious_matrix.png in outputs/.
- Tests: 23 passing (model shapes/CBAM count, hand-computed metrics,
  augmentation one-hot invariance, plus the Phase 0 suite).
- `/code-review` findings applied: decision threshold lives once in
  `metrics.DEFAULT_THRESHOLD`; evaluate restores `input_size` (not just
  arch/cbam) from the checkpoint; first epoch always checkpoints (no stale
  best.pt after a diverged run); `backbone_ckpt_path` anchored to repo root
  like other paths; source-checkpoint `fc.*` keys dropped before backbone
  load (9-class vs 8-logit shape clash); `--pretrained/--no-pretrained`,
  `--cbam/--no-cbam` both directions on CLI.
- Smoke run (1 epoch, 4090): pipeline converges — val macro-F1 0.9473 after
  a single epoch; GAN-synthesized data is learnable fast, expect the full
  run to saturate early. Smoke run already shows Scratch as the dominant
  spurious label inside mixes (Edge-Ring→+Scratch 0.43) — watch whether
  that persists at convergence.

**To run on the 5090 (own terminal, then paste final metrics here):**
```bash
cd wafer-mixed && source ../.venv/bin/activate
python -m wafer_mixed.train                 # batch 128 default, ~30 epochs w/ early stop
python -m wafer_mixed.evaluate              # prints tables; artifacts → outputs/
```

**Metrics table (fill from the 5090 run):** _pending_

**Next (Phase 2, fresh session, after metrics land here):** transfer study —
3 arms (scratch / WM-811K supervised init / wafer-ssl SimCLR init) via
`backbone_ckpt_path`, same budget + seeds, results → docs/TRANSFER.md.
