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
python -m wafer_mixed.train
Device: cuda  |  arch: resnet18  |  cbam: True  |  pretrained: False
Loading data...
  Labels: ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Scratch', 'Random']
Loss: BCEWithLogitsLoss (8 independent labels, no pos_weight — rare labels verified learnable without it; see module docstring)
Epoch   1/30  train loss 0.1223 f1 0.8261  |  val loss 0.1344 f1 0.9324 *
Epoch   2/30  train loss 0.0285 f1 0.9748  |  val loss 0.0233 f1 0.9829 *
Epoch   3/30  train loss 0.0258 f1 0.9822  |  val loss 0.0233 f1 0.9762
Epoch   4/30  train loss 0.0183 f1 0.9764  |  val loss 0.0264 f1 0.9811
Epoch   5/30  train loss 0.0171 f1 0.9823  |  val loss 0.0316 f1 0.9691
Epoch   6/30  train loss 0.0143 f1 0.9852  |  val loss 0.0168 f1 0.9855 *
Epoch   7/30  train loss 0.0138 f1 0.9865  |  val loss 0.0169 f1 0.9906 *
Epoch   8/30  train loss 0.0124 f1 0.9811  |  val loss 0.0212 f1 0.9768
Epoch   9/30  train loss 0.0112 f1 0.9869  |  val loss 0.0142 f1 0.9870
Epoch  10/30  train loss 0.0108 f1 0.9903  |  val loss 0.0190 f1 0.9838
Epoch  11/30  train loss 0.0096 f1 0.9885  |  val loss 0.0141 f1 0.9629
Epoch  12/30  train loss 0.0096 f1 0.9894  |  val loss 0.0153 f1 0.9854
Epoch  13/30  train loss 0.0095 f1 0.9867  |  val loss 0.0177 f1 0.9840
Epoch  14/30  train loss 0.0086 f1 0.9880  |  val loss 0.0120 f1 0.9868
Early stop: no val macro-F1 gain for 7 epochs.

Done. Best val macro-F1 (8 labels @0.5): 0.9906
Checkpoint : /home/alex8642/wafer-classifier/wafer-mixed/outputs/best.pt
**Note for Claude in next phase - a bug had to be fixed in src/wafer_mixed/train.py --> all_preds to detach before converting for numpy**

**Metrics table (fill from the 5090 run):**
python -m wafer_mixed.evaluate
Checkpoint : /home/alex8642/wafer-classifier/wafer-mixed/outputs/best.pt  (epoch 7, val macro-F1 0.9906)
Evaluating: 100%|███████████████████████████████████████████████████████████████████████| 60/60 [00:03<00:00, 19.40it/s]

================================================================
TEST SET RESULTS  (multi-label @ sigmoid 0.5)
================================================================
  Macro-F1 (8 labels) : 0.9846  ← headline metric
  Exact-match ratio   : 0.9696  (all 8 labels correct)

  label         prec  recall      f1  support
  Center      1.0000  0.9996  0.9998     2600
  Donut       1.0000  1.0000  1.0000     2400
  Edge-Loc    0.9984  0.9769  0.9876     2600
  Edge-Ring   0.9925  0.9967  0.9946     2400
  Loc         0.9994  0.9778  0.9885     3600
  Near-full   0.9643  0.9000  0.9310       30  (small support)
  Scratch     0.9984  0.9805  0.9894     3800
  Random      0.9828  0.9884  0.9856      173  (small support)

  Subset breakdown (macro-F1 averages only labels with support in the subset):
  subset        n  exact-match   macro-F1
  normal      200       1.0000          —
  single     1403       0.9879     0.9849
  mixed      6000       0.9643     0.9933

  Per-label recall, single vs mixed (— = label never in subset):
  label        single    mixed
  Center       1.0000   0.9996
  Donut        1.0000   1.0000
  Edge-Loc     0.9800   0.9767
  Edge-Ring    0.9950   0.9968
  Loc          0.9700   0.9782
  Near-full    0.9000        —
  Scratch      0.9950   0.9797
  Random       0.9884        —

  Top spurious activations inside mixes (true label → falsely predicted label):
    Edge-Loc → +Edge-Ring: 0.007
    Donut → +Edge-Ring: 0.006
    Scratch → +Edge-Ring: 0.005
    Donut → +Scratch: 0.003
    Center → +Scratch: 0.003
    Loc → +Scratch: 0.003

Per-label CSV : /home/alex8642/wafer-classifier/wafer-mixed/outputs/per_label_metrics.csv
Metrics JSON  : /home/alex8642/wafer-classifier/wafer-mixed/outputs/metrics.json
Spurious matrix: /home/alex8642/wafer-classifier/wafer-mixed/outputs/spurious_matrix.png

**Next (Phase 2, fresh session, after metrics land here):** transfer study —
3 arms (scratch / WM-811K supervised init / wafer-ssl SimCLR init) via
`backbone_ckpt_path`, same budget + seeds, results → docs/TRANSFER.md.
