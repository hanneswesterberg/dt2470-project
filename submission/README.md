# Automatic chord recognition

This folder is the complete DT2470 project submission. It contains the trained
hierarchical CNN/HSMM model, fixed 15-song CNN-only RWC evaluation data, example
audio, generated results, and the presentation notebook.

## Quick start

Run all commands from this `submission/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab project_presentation.ipynb
```

Then use **Run All Cells**. The notebook reproduces the CNN-only evaluation,
compares CNN-only with CNN+HSMM across all 15 continuous test songs, and runs
chord recognition on the selected song. Compact saved CNN posteriors make the
full HSMM evaluation reproducible without distributing all 15 WAV files. Its
final section contains a synchronized audio player with rolling chord cards.

If the notebook is already open in VS Code or Jupyter with another Python
kernel, you can instead run only its first setup cell. That cell creates the
same `.venv` and registers a **Python (DT2470 submission)** kernel. Select the
new kernel, then run all cells.

The default example can also be run directly:

```bash
python main.py \
  --source local \
  --input-dir audio \
  --cnn-checkpoint models/best_model.pt \
  --language-model models/harmonic_model.pkl \
  --split-manifest models/split_manifest.json \
  --song beatles_something \
  --non-interactive
```

To reproduce only the evaluation:

```bash
python training/evaluate_rwc.py \
  --feature-dir features \
  --checkpoint models/best_model.pt
```

See `ARCHITECTURE.md` for the model design and musical motivation.
