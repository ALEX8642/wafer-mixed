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
