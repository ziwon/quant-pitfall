# Repository Guidelines

## Project Structure & Module Organization

This Python 3.10+/PyTorch project teaches nine neural-network quantization pitfalls through reproducible experiments.

- `src/quant_pitfalls/`: quantizers and observers in `quant/`, Conv–BN folding in `layers/`, integer inference in `sim/`, and CIFAR-10 models in `models/`.
- `tests/`: numerical expectations and behavioral regression tests.
- `notebooks/`: numbered lessons, such as `01_clipping-mse.ipynb`.
- `train.py`: float, PTQ, and QAT training CLI. `scripts/reproduce.sh` runs tests, notebooks, figures, and result collection.
- `docs/README.md`: learning guide; `docs/reproduction.md`: execution modes; `docs/evidence.md`: numerical provenance. Save SVG figures under `docs/figures/`.

## Build, Test, and Development Commands

Use uv for dependencies. Run commands from the repository root.

```bash
uv sync --frozen --all-groups
uv run pytest -q
uv run pytest tests/test_rounding.py -q
uv run jupyter lab notebooks/
bash scripts/reproduce.sh --from-results
uv build
```

The reproduction command recomputes synthetic experiments and renders recorded training measurements. `--skip-training` requires complete runs and checkpoints; `--train` runs or resumes training and integer evaluation. Consult the reproduction guide before using either mode.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and typed public signatures. Explain formulas, dtype behavior, and rounding rules in docstrings. Preserve public APIs and result formats. No formatter or linter is configured. Keep code, comments, and plot labels in English; write learning documentation in Korean.

## Testing Guidelines

Use pytest with `tests/test_<feature>.py` files and `test_<behavior>` functions. Seed randomized inputs and use explicit numerical tolerances. Cover rounding ties, clipping, gradients, BN freezing, and overflow. Preserve expected values and tolerances unless an independent numerical reference justifies a correction. Run focused tests during development and the full reproduction pipeline for changes affecting recorded lessons.

## Commit & Pull Request Guidelines

Follow `CONTRIBUTING.md`. Use concise imperative commit subjects. PRs should explain the problem, include reproduction commands and validation results, and link relevant issues. Include updated figures for experiment changes. Commit or publish only when explicitly requested.

## Reproducibility & Generated Files

Change one variable per comparison; record seeds, environment, sample counts, and runtime. Ground reported numbers in tests, notebooks, and archived measurements. Keep datasets, checkpoints, and logs in ignored `data/` and `runs/`. Separate recorded accuracy from new measurements.
