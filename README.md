# DT2470 chord recognition

This is our group project for DT2470. It estimates chords from an audio file and writes a timestamped chord sequence.

The default system is a hierarchical CNN followed by a second-order HSMM. An optional Gaussian HMM pipeline is included for comparison.

## Run it

Set up the Python environment:

```bash
python3 setup_env.py
```

Put an MP3, WAV, FLAC, or OGG file in `input/`, then run:

```bash
.venv/bin/python main.py
```

Choose a song from the list. The results are written to `output/`:

```text
output/
├── <song>_viewer.html
├── media/                              Browser-compatible audio when needed
├── hierarchical_cnn_hsmm/
│   ├── <song>_chords.csv
│   └── <song>_chords.json
└── summary.json
```

Open `<song>_viewer.html` in a browser to listen to the song while following the predicted chords. The chord blocks can also be clicked to seek in the audio.

A specific song can be run without the menu:

```bash
.venv/bin/python main.py --non-interactive --source local --song example.mp3
```

The main options are:

- `--pipeline hierarchical` or `--pipeline gaussian`
- `--device cpu`, `--device mps`, or `--device cuda`
- `--hsmm-order 1` or `--hsmm-order 2`
- `--top-roots 4` controls how many root candidates are passed from the CNN to the HSMM

## Repository structure

```text
dt2470-project/
├── main.py                         Runs inference
├── train_hierarchical_cnn.py       Trains the CNN
├── train_gaussian_baseline.py      Trains the Gaussian baseline
├── setup_env.py                    Creates the local Python environment
├── requirements.txt
│
├── models/
│   ├── hierarchical/               CNN checkpoint, HSMM model, and data split
│   └── gaussian_hmm_10.pkl         Gaussian baseline checkpoint
│
├── input/                          Audio used for inference
├── output/                         Generated results and HTML viewers
├── data/                           Data loading and annotation code
├── tests/                          Unit and integration tests
│
├── src/
│   ├── hierarchical/
│   │   ├── cqt_frontend.py         Audio preprocessing and CQT windows
│   │   ├── hierarchical_cnn.py     CNN architecture
│   │   ├── hsmm_decoder.py         HSMM and Viterbi decoding
│   │   ├── inference_pipeline.py   Complete CNN + HSMM pipeline
│   │   ├── vocabulary.py           Chord labels and state indices
│   │   └── artifacts.py            Model loading and validation
│   ├── features/                   Features for the Gaussian baseline
│   ├── hmm/                        Baseline HMM and Viterbi code
│   ├── models/                     Baseline Gaussian model code
│   ├── inference/                  Shared inference helpers
│   ├── evaluation/                 Evaluation code
│   └── visualization/              Plots and the interactive viewer
│
└── notebooks/
    ├── architecture_notes.ipynb
    └── TinyAAM_BeatNet_Gaussian_HMM_Full_Pipeline.ipynb
```

The files under `src/hierarchical/` contain the default system. The files under `src/features/`, `src/hmm/`, and `src/models/` support the optional Gaussian comparison model.

## Architecture

### Musical view

The CNN looks at the sound of a short time window and describes the chord it hears. It is trained with four musical targets:

```text
root + quality + seventh + extensions
```

For example, a label can be understood as a root such as C, a quality such as major or minor, a seventh such as major seventh, and any additional chord tones such as a ninth. The targets are predicted hierarchically: the root is predicted first, then the quality relative to that root, followed by the seventh and extensions. This reflects how musicians name chords and lets the same quality pattern be recognized in different keys.

The current HSMM state space contains root, quality, and seventh (12 x 6 x 4 = 288 states). The extension head is trained and its probabilities are available from the CNN, but extensions are not yet part of the decoded chord names or the HSMM state space.

### 1. Audio and CQT features

`src/hierarchical/cqt_frontend.py` loads the audio as mono at 22,050 Hz and extracts its harmonic part. It then calculates a constant-Q transform with 216 bins and combines every three adjacent bins, producing 72 frequency features.

The CNN receives windows with shape `72 x 32`. A window is produced every 200 ms and covers about 0.74 seconds of audio.

### 2. Hierarchical CNN

`src/hierarchical/hierarchical_cnn.py` predicts a chord in three steps:

```text
root -> quality given the root -> seventh given the root and quality
```

There are 12 roots, 6 qualities, and 4 seventh classes. This gives 288 possible root-quality-seventh chords. Before predicting quality, the CQT is rotated relative to each possible root. This lets the quality part of the network learn the same chord shape in every key.

The model also has an extension head. It is trained from multi-hot extension labels, so a chord can have more than one extension at the same time. The current inference output does not include those extensions yet because the decoder uses the 288-state root-quality-seventh vocabulary.

### 3. HSMM and Viterbi decoding

The CNN makes a prediction every 200 ms. These frame-level predictions change too often to be used directly as chord events.

`src/hierarchical/hsmm_decoder.py` uses a hidden semi-Markov model to choose the chord sequence and its boundaries. It receives the rolling sequence of CNN emission probabilities. For every possible chord segment, it sums the emission evidence over the frames in that segment, then combines it with:

- learned chord durations; and
- harmonic transition probabilities.

The default second-order model remembers the two previous chord events. Musically, its transition model mainly learns the movement of the root in semitones, together with the quality and seventh of the new chord. For example, it can learn that a dominant-seventh chord often moves down a fifth to a new root, or that a minor chord tends to follow a particular relative root movement. The model stores root movements relative to the previous root, so the same progression can be used in different keys. It also learns how long chords normally last.

The HSMM therefore smooths the frame-level CNN predictions into musical events: it chooses a chord, decides when that chord ends, checks the likely duration, and scores the transition from the previous two chords. Decoding is done with segmental Viterbi and beam pruning.

### 4. Output

`src/hierarchical/inference_pipeline.py` connects preprocessing, the CNN, and the HSMM. CNN predictions are cached in `.cache/chordrec/` so changing the decoder does not require recalculating the CQT and CNN output.

`src/visualization/interactive_viewer.py` creates the HTML player in `output/`.

## Models and data

The files needed for inference are already in `models/hierarchical/`:

```text
best_model.pt          CNN weights and chord vocabulary
harmonic_model.pkl     Transition and duration model for the HSMM
split_manifest.json    Fixed 70/15/15 train, validation, and test split
```

Running inference does not require Google Drive access. The large training feature files and the original audio dataset are not stored in Git. This keeps the repository small and avoids distributing copyrighted audio.

The shared project data is kept in Google Drive under `Delas med mig/DT2470 Project`. The relevant folders are:

```text
DT2470 Project/
├── Chord Recognition - Models, Data & Evaluation/
│   ├── CNN Features (100 RWC songs)/
│   ├── Checkpoints and Split/
│   ├── Rolling CNN Predictions/
│   └── HSMM Evaluation Results/
└── full_data/
    ├── 0001-1000-audio-mixes/
    └── 0001-1000-annotations-v1.1.0/
```

`data/load_data.py` looks for this shared folder through Google Drive Desktop. If it is not found automatically, set `AAM_DRIVE_DIR` in `.env` to the local folder path. The code does not use a separate personal copy of the dataset.

## Training the CNN

The CNN training implementation is `train_hierarchical_cnn.py`. It uses the CNN implementation from `src/hierarchical/hierarchical_cnn.py`, so training and inference do not have separate copies of the architecture.

The script expects one cached feature file per song:

```text
data/features_cqt72_v1/RWC_P001.npz
data/features_cqt72_v1/RWC_P002.npz
...
```

Each file must contain `X`, `root`, `quality`, `seventh`, and multi-hot extension targets. `X` must have shape `(N, 72, 32)`. Run training with:

```bash
.venv/bin/python train_hierarchical_cnn.py \
  --feature-dir /path/to/features_cqt72_v1
```

The script uses the split in `models/hierarchical/split_manifest.json` and only loads the training and validation songs. New checkpoints and training logs are written to `models/training_runs/hierarchical_cnn/`, which is ignored by Git.

The training settings are defined in the script and can be changed with its command-line options.

The script automatically looks for an extension target key named `extensions`, `extension`, `ext`, `extension_targets`, or `ext_targets`. If the data uses another key, pass it explicitly:

```bash
.venv/bin/python train_hierarchical_cnn.py \
  --feature-dir /path/to/features_cqt72_v1 \
  --extension-key extensions
```

The Gaussian baseline can be trained separately:

```bash
.venv/bin/python train_gaussian_baseline.py
```

## Tests

Run all tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The architecture notes are in `notebooks/architecture_notes.ipynb`.

## RWC test evaluation

The feature-level evaluation command measures the CNN on the fixed 15-song test split:

```bash
.venv/bin/python evaluate_rwc.py \
  --feature-dir "/path/to/DT2470 Project/Chord Recognition - Models, Data & Evaluation/CNN Features (100 RWC songs)"
```

The test songs are `37, 34, 90, 89, 3, 60, 72, 17, 35, 12, 69, 75, 40, 47, 39`. The report contains root accuracy, root-plus-quality accuracy, and root-plus-quality-plus-seventh accuracy. It also reports the percentage of each song's annotated duration for which the predicted chord is correct. The feature files contain the rolling CNN inputs and labels, so this command evaluates the CNN emissions directly. Full CNN + HSMM evaluation is available through `evaluate_hierarchical.py` when the corresponding audio and annotations are available.

## Evaluating against annotations

For a larger experiment, use `evaluate_hierarchical.py` instead of selecting songs one at a time. It finds matching audio and `*_beatinfo.arff` annotation files, runs the hierarchical CNN/HSMM on every pair, and writes one result per song plus aggregate averages.

The evaluator compares the predicted root and major/minor quality against the existing AAM annotations. Seventh and extension labels are kept in the model output, but the current AAM evaluator does not score them because the annotation parser exposes the older triad labels.

```bash
.venv/bin/python evaluate_hierarchical.py \
  --audio-dir /path/to/audio \
  --annotation-dir /path/to/annotations
```

Results are written to `output/evaluation/`:

```text
results.json       Full per-song results
results.csv        One row per song
aggregate.json     Mean accuracy, coverage, and beat metrics
```

The evaluator uses the same cache as normal inference. The first run extracts the audio features and CNN emissions; later runs can reuse them. Use `--limit 10` for a quick check before starting a full collection run.

## Notebook

[`notebooks/Chord_Recognition_Workbench.ipynb`](notebooks/Chord_Recognition_Workbench.ipynb)
provides an interactive frontend for local audio inference, prediction
inspection, optional model training, annotated-dataset evaluation, and the RWC
feature-level CNN benchmark. It calls the same modules and workflows as the
command-line tools; it does not duplicate the model implementation. Set up the
project environment with `python3 setup_env.py`, select that environment as the
notebook kernel, and place local audio files in `input/`. Training and
full-dataset evaluation are disabled by default and require the corresponding
external datasets.