# Automatic chord recognition

This repository is the complete DT2470 submission. It contains the trained
hierarchical CNN and second-order HSMM, the fixed 15-song RWC evaluation data,
continuous CNN posteriors, example audio, generated results, and two presentation
notebooks.

## Submitted notebook: Google Colab

[`project_presentation.ipynb`](project_presentation.ipynb) is the primary
hand-in. Its cells are already executed so the evaluation, plots, and player are
visible immediately. To reproduce the results, open it in Google Colab and
choose **Runtime → Run all**.

The Colab setup downloads this repository's `main` branch into a temporary
runtime. It requires internet access but does not require Google Drive, manual
model uploads, or a virtual environment. It evaluates CNN-only and CNN+HSMM on
all 15 continuous test songs and uses `RWC_P039` as the default playable example.

## Local notebook

[`project_presentation_local.ipynb`](project_presentation_local.ipynb) contains
the same implementation for JupyterLab or VS Code. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab project_presentation_local.ipynb
```

Alternatively, run only the notebook's setup cell first. It creates `.venv` and
registers a **Python (DT2470 submission)** kernel. Select that kernel and then
run all cells.

## Command-line execution

Run the default demonstration directly:

```bash
python main.py \
  --source local \
  --input-dir audio \
  --cnn-checkpoint models/best_model.pt \
  --language-model models/harmonic_model.pkl \
  --split-manifest models/split_manifest.json \
  --song RWC_P039 \
  --non-interactive
```

Reproduce only the evaluation:

```bash
python training/evaluate_rwc.py \
  --feature-dir features \
  --checkpoint models/best_model.pt
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the model design and musical
motivation.
