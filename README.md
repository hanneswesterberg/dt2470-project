# DT2470 automatic chord recognition

This repository implements the workflow in
`notebooks/TinyAAM_BeatNet_Gaussian_HMM_Full_Pipeline.ipynb` as reusable Python
modules and two interactive terminal entry points:

- `train.py` trains and saves models or individual HMM components.
- `main.py` loads complete models and runs inference/evaluation only.

The current model recognizes 24 states (12 major and 12 minor chords). It uses
beat-synchronous 24D bass/midrange CQT chroma, shared major/minor Gaussian
emissions, transposition-invariant HMM transitions, and weighted Viterbi
decoding. Unsupported reference chords are marked `N` and excluded from chord
accuracy.

## Setup and dependencies

Python 3.10 or newer is recommended for the main application. You do not need
to create or activate environments manually. Running either entry point creates
and updates `.venv` automatically, then restarts inside it:

```bash
python3 train.py
python3 main.py
```

BeatNet has an incompatible legacy dependency stack, so it is kept in a second
environment, `.venv-beatnet`, using Python 3.9. Selecting BeatNet in the
inference wizard creates that environment automatically. To prepare both
environments ahead of time, run:

```bash
python3 setup_env.py --with-beatnet
```

On macOS the setup also installs PortAudio with Homebrew when PyAudio needs it.
Dependency changes are detected from the requirements files and installed on
the next launch. Both environments are ignored by Git.

Convenience commands are also available:

```bash
make setup   # prepare .venv and .venv-beatnet
make train   # open the training wizard
make run     # open the inference wizard
make test
```

Configure the mounted Google Drive dataset in an untracked `.env` file:

```dotenv
GOOGLE_DRIVE_EMAIL=your-address@example.com
AAM_PROJECT_FOLDER=path/to/project/folder
```

On macOS, the code also auto-detects a uniquely mounted Google Drive containing
the shared `DT2470 Project` folder, so `.env` can be omitted in that setup. Use
`AAM_DRIVE_DIR=/absolute/path/to/project` for a direct cross-platform override.

The default dataset layout under that folder is:

```text
full_data/
├── 0001-1000-audio-mixes/
└── 0001-1000-annotations-v1.1.0/
```

Only paths into the mounted Drive are used; the full audio dataset is not
copied into this repository.

## Train models

Start the guided training wizard:

```bash
python3 train.py
```

It asks:

1. Whether to train a complete model, only the Gaussian emission component, or
   only the HMM transition component.
2. How many randomly selected AAM songs to use.
3. What to name the saved artifact.
4. Whether to start after showing the final configuration.

Artifacts are organized as:

```text
models/
├── gaussian_hmm_80.pkl              # complete, runnable model
├── emissions/
│   └── emission_experiment.pkl      # component-only experiment
└── transitions/
    └── transition_experiment.pkl    # component-only experiment
```

Only complete models directly under `models/` can be selected by `main.py`.
Component-only artifacts are retained for experiments and future model
assembly.

For repeatable, non-interactive training:

```bash
python3 train.py \
  --non-interactive \
  --target full \
  --model gaussian_hmm_80 \
  --training-songs 80
```

Use `--target emission` or `--target hmm` for individual components. Add
`--overwrite` to replace an existing artifact.

## Run inference

Start the guided inference wizard:

```bash
python3 main.py
```

It:

1. Finds songs stored locally in `input/` and songs available in Google Drive.
2. Asks whether to use local songs, cloud songs, or both.
3. Lets you choose exact local files, Drive IDs, a random Drive sample, or all.
4. Lets you select one or several complete models.
5. Asks whether to compare against ground truth, when annotations are available.
6. Asks which beat tracker to use.

When several models are selected, beat detection and CQT extraction run once
per song and the prepared features are decoded by every model.

Each model's predicted timeline is written to
`output/<model-name>/<song-name>_chords.csv`. In addition, every song gets a
visual chord chart at `output/charts/<song-name>_chord_chart.png`. The chart
groups consecutive beat intervals into bars of four and shows all selected
models together. These are visual four-beat groups beginning at the first
detected or annotated beat; they do not claim to detect musical downbeats.

A machine-readable run summary is written to `output/summary.json`.

To evaluate an input song, place its annotation beside it using either of these
names:

```text
input/example.wav
input/example_beatinfo.arff  # preferred
input/example.arff           # also supported
```

An `input/annotations/example_beatinfo.arff` layout is also supported. When an
annotation is found and comparison is selected in the wizard, `main.py`
reports time-aligned chord accuracy, prediction coverage, and beat
precision/recall/F1. The chart then uses annotated beats as its grid, includes
a ground-truth row, and marks predictions green or red for correct or incorrect
chords. Without comparison, the chart contains prediction rows only.

Useful options:

```bash
python3 main.py --help
python3 main.py --list-models
python3 main.py --model gaussian_hmm_80 --source local
python3 main.py --model model_a --model model_b --source cloud --song 0001 --song 0002
python3 main.py --non-interactive --model gaussian_hmm_80 --source local --compare-ground-truth
```

Non-interactive runs do not compare annotations unless
`--compare-ground-truth` is supplied. Use `--no-compare-ground-truth` to
explicitly disable comparison in either mode.

## Beat tracking

`--beat-tracker auto` is the default. It uses BeatNet when the `BeatNet` package
is available and otherwise uses librosa, which is part of the normal
requirements. BeatNet runs through an isolated Python 3.9 subprocess; its
NumPy, madmom, and PyTorch versions never mix with the main environment.
Selecting `beatnet` explicitly installs or updates that environment if needed.

## Module boundaries

- `data/load_data.py`: Google Drive and local input discovery
- `data/annotations.py`: ARFF parsing and chord normalization
- `src/features/`: audio-to-feature transformations such as CQT chroma
- `src/music/`: the shared chord vocabulary and label conversions
- `src/beat_net/`: beat-tracker protocol and available implementations
- `src/hmm/`: Gaussian emissions, relative transitions, and Viterbi decoding
- `src/models/`: composed recognition models and checkpoint handling
- `src/training/`: component training and the complete dataset workflow
- `src/inference/`: audio-to-chord pipeline and prediction timelines
- `src/evaluation/`: time-aligned chord and beat metrics
- `src/visualization/`: four-beat-bar PNG charts for predictions and references
- `src/cli/`: validated terminal prompts shared by `train.py` and `main.py`

Each package exports its intended public API from `__init__.py`. For example:

```python
from src.features import CQTChromaExtractor
from src.training import (
    train_emission_model,
    train_transition_model,
    train_hmm,
    train_model,
)
from src.inference import ChordRecognitionPipeline
```

The emission and transition components can be trained independently for
experiments:

```python
emission = train_emission_model(features, states)
transition = train_transition_model(song_state_sequences, smoothing=0.5)
model = train_hmm(features, states, song_state_sequences)
```

The interfaces between these modules are deliberately narrow. Alternative beat
trackers, feature representations, emission distributions, transition models,
or decoders can be introduced without rewriting `main.py`.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
