# wafer-mixed

Multi-label robustness study on **MixedWM38** — extending the
[wafer-defect-classifier](https://github.com/ALEX8642/wafer-defect-classifier)
pipeline from single-label WM-811K to *mixed* (superposed) defect patterns,
the production-realistic case. Satellite repo, same pattern as
[wafer-ssl](https://github.com/ALEX8642/wafer-ssl).

**Status:** Phase 0 complete (scaffold + data pipeline). See `STATUS.md` for
the session-by-session log and `PLAN` phases in the workspace plan doc.

## Data

MixedWM38 (Wang et al. 2020): 38,015 wafer maps, 52×52, 38 pattern types
that decompose into **8 basic defect labels** → framed as 8-way multi-label
classification, not 38-way multi-class. Facts verified locally; see
[docs/DATA.md](docs/DATA.md) for the frequency tables, sample grids, and the
label-ordering verification.

**Data boundary:** this repo uses only the public MixedWM38 dataset — no
proprietary or employer data appears here, in any branch, at any point.

## Quickstart (Phase 0)

```bash
pip install -e . -r requirements.txt
python scripts/download_data.py     # ~412 MB download + verify + write splits
python scripts/eda.py               # regenerate docs/DATA.md + assets/
pytest                              # includes split-leakage checks
```

## License

MIT — see [LICENSE](LICENSE).
