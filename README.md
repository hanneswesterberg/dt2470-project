# Automatic Chord Recognition with a Hierarchical CNN and HSMM

This repository is the DT2470 group project for recognizing chords directly
from audio. The main system uses a hierarchical convolutional neural network
(CNN) to estimate chord roots, qualities, and sevenths, followed by a
second-order hidden semi-Markov model (HSMM) that produces a stable timeline of
chord events.

The repository also retains our earlier Gaussian-HMM system as an optional
baseline. The hierarchical CNN and HSMM are the default implementation.

## Quick start

### 1. Prepare the environment

Python 3.10 or newer is recommended. From the repository root, run:

```bash
python3 setup_env.py
```

This creates `.venv`, installs the required packages, and keeps dependencies
isolated from the rest of the computer.

### 2. Run chord recognition

Several local songs can be selected through the interactive interface:

```bash
python3 main.py
```

The program lists the audio files in `input/`. Enter a number, a range such as
`1,3-5`, or `all`.

To run a specific local file without the interactive questions:

```bash
python3 main.py \
  --non-interactive \
  --source local \
  --song hey_jude.mp3
```

The hierarchical system is the default. It can also be selected explicitly:

```bash
python3 main.py \
  --pipeline hierarchical \
  --source local \
  --top-roots 4 \
  --hsmm-order 2
```

Useful options:

- `--top-roots 4` evaluates the four most likely CNN root candidates. Use
  `--top-roots 12` for exhaustive root enumeration.
- `--hsmm-order 2` uses the default second-order harmonic model. Use `1` for
  the first-order comparison.
- `--device cpu`, `--device mps`, or `--device cuda` selects the PyTorch device.
- `--cache-dir PATH` changes where continuous CNN predictions are cached.

### 3. Find the results

The default pipeline writes:

```text
output/
├── <song>_viewer.html
├── hierarchical_cnn_hsmm/
│   ├── <song>_chords.csv
│   └── <song>_chords.json
└── summary.json
```

Open `<song>_viewer.html` in a browser to hear the audio with a synchronized
chord display and clickable timeline. The CSV and JSON files contain the same
events as structured data. Generated output is ignored by Git.

## How the system works

The complete inference path is:

```text
Audio
  ↓ mono, 22,050 Hz and harmonic separation
Continuous CQT
  ↓ 216 frequency bins grouped into 72 features
Overlapping 72 × 32 windows
  ↓ one observation every 200 ms
Hierarchical CNN
  ├── P(root | audio)                         12 roots
  ├── P(quality | root, audio)                 6 qualities
  └── P(seventh | root, quality, audio)        4 seventh classes
  ↓ joint posterior over 12 × 6 × 4 = 288 chord states
Second-order HSMM
  ├── CNN acoustic evidence
  ├── duration probabilities
  └── transitions conditioned on two previous chord events
  ↓
Timestamped chord timeline
```

### CQT72 audio frontend

Implementation: `src/hierarchical/cqt_frontend.py`

Audio is loaded as mono at 22,050 Hz and separated with harmonic-percussive
source separation. A constant-Q transform is calculated with 36 bins per
octave and 216 total bins beginning at C1. Groups of three adjacent bins are
summed to create 72 frequency features.

The continuous representation is sampled every 200 ms. Each CNN observation
contains 32 interpolated CQT frames, giving the network approximately 0.74
seconds of context. Square-root compression and per-frame normalization match
the preprocessing used by the supplied checkpoint.

### Hierarchical CNN

Implementation: `src/hierarchical/hierarchical_cnn.py`

The CNN is not a flat 288-class classifier. It represents a chord as three
dependent decisions:

```text
root → quality conditioned on root → seventh conditioned on root and quality
```

The root head predicts one of 12 pitch classes. For every candidate root, the
72 CQT features are rotated into a root-relative representation. A second
encoder then predicts one of these qualities:

```text
major, minor, diminished, augmented, suspended 2, suspended 4
```

The seventh head predicts:

```text
none, flat seventh, major seventh, diminished seventh
```

During inference, the system combines the conditional probabilities:

```text
P(root, quality, seventh | audio)
  = P(root | audio)
  × P(quality | root, audio)
  × P(seventh | root, quality, audio)
```

This preserves alternative chord candidates for the temporal decoder instead
of greedily choosing each head independently. The checkpoint also contains an
extension head, but extensions are not decoded because the available extension
training labels were too imbalanced.

### Harmonic language model and HSMM

Implementation: `src/hierarchical/hsmm_decoder.py`

The CNN produces an observation every 200 ms, but chords normally persist over
several observations. The HSMM jointly decides chord identities and event
boundaries.

Each candidate segment is scored using:

- accumulated CNN evidence across the segment;
- a learned chord-duration distribution; and
- a transposition-invariant harmonic transition probability.

The second-order model considers the two previous chord events when scoring a
new event. Transitions are expressed as relative root movements, allowing a
progression learned in one key to apply in another key.

Decoding uses beam-pruned segmental Viterbi. It is an approximate decoder: a
beam retains the strongest hypotheses, and only the most acoustically likely
states are expanded for each segment. The default parameters reproduce the
settings documented by the research notebook.

### End-to-end inference and caching

Implementation: `src/hierarchical/inference_pipeline.py`

This module connects the frontend, CNN, and HSMM. Continuous CNN posteriors are
cached under `.cache/chordrec/` because CQT extraction and CNN inference are the
expensive parts of the pipeline.

Cache keys include the audio hash, CNN checkpoint hash, frontend configuration,
root-pruning setting, and pipeline version. A cached prediction is therefore
not accidentally reused after changing the audio or model.

## Repository guide

```text
dt2470-project/
├── main.py                         Main inference command
├── setup_env.py                    Creates and updates local environments
├── train_gaussian_baseline.py      Trains the earlier Gaussian-HMM baseline
├── requirements.txt                Main Python dependencies
├── requirements-beatnet.txt        Isolated legacy BeatNet dependencies
│
├── models/
│   ├── hierarchical/               CNN, harmonic model, and split manifest
│   └── gaussian_hmm_10.pkl         Example Gaussian baseline
│
├── input/                          Local audio supplied to main.py
├── output/                         Generated timelines and charts
│
├── src/
│   ├── hierarchical/
│   │   ├── cqt_frontend.py         Audio → rolling CQT72 windows
│   │   ├── hierarchical_cnn.py     CNN architecture and joint probabilities
│   │   ├── hsmm_decoder.py         Harmonic model and segmental Viterbi
│   │   ├── inference_pipeline.py   End-to-end inference and caching
│   │   ├── vocabulary.py           288-state chord representation
│   │   └── artifacts.py            Model artifact discovery and validation
│   │
│   ├── features/                   Gaussian baseline features
│   ├── hmm/                        Gaussian emissions and HMM decoding
│   ├── models/                     Gaussian baseline model serialization
│   ├── inference/                  Shared timelines and baseline pipeline
│   ├── evaluation/                 Timeline evaluation utilities
│   ├── visualization/              Interactive player and chord charts
│   └── beat_net/                   Optional BeatNet integration
│
├── data/                           Audio discovery and AAM annotations
├── tests/                          Unit and integration tests
└── notebooks/
    ├── chord_recognition_architecture_handoff.ipynb
    └── TinyAAM_BeatNet_Gaussian_HMM_Full_Pipeline.ipynb
```

The hierarchical package is the main implementation. The older modules under
`src/features`, `src/hmm`, and `src/models` support the Gaussian baseline and
are retained for comparison with the final architecture.

## Model files

The default inference artifacts live in `models/hierarchical/`:

```text
best_model.pt          Hierarchical CNN parameters and vocabulary
harmonic_model.pkl     Transition counts, priors, and duration samples
split_manifest.json    Fixed 70/15/15 RWC train/validation/test split
```

These files are sufficient for local inference; Google Drive access is not
required. PyTorch and pickle files should only be loaded from trusted sources.

The chord vocabulary contains 12 roots, six qualities, and four seventh
classes, giving 288 states. Chord extensions and inversions are not part of the
default decoder, and the system currently has no explicit no-chord state.

## Data and training

Large training datasets and copyrighted recordings are not required for
inference and should not be committed to the repository. Users can place their
own MP3, WAV, FLAC, or OGG files in `input/`.

The supplied CNN and harmonic model use the fixed seed-42 RWC split:

- 70 training songs;
- 15 validation songs; and
- 15 held-out test songs.

The research handoff did not contain the original authoritative CNN training
loop. This repository therefore provides reproducible inference from the saved
checkpoint without claiming that the training experiment can be recreated
exactly. The harmonic transition model must be trained on sequential song
annotations, not isolated-chord datasets.

`train_gaussian_baseline.py` only trains the earlier Gaussian-HMM baseline:

```bash
python3 train_gaussian_baseline.py
```

## Gaussian-HMM baseline

The baseline recognizes 24 major/minor states using beat-synchronous chroma,
Gaussian emissions, a relative first-order transition model, and ordinary
Viterbi decoding.

Run it with:

```bash
python3 main.py \
  --pipeline gaussian \
  --non-interactive \
  --model gaussian_hmm_10 \
  --source local
```

BeatNet has an older dependency stack and therefore uses a separate Python 3.9
environment. Prepare it only when needed:

```bash
python3 setup_env.py --with-beatnet
```

## Tests

Run the complete test suite with:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover the hierarchical vocabulary and CQT windows, model artifact
handling, timeline export, Gaussian components, evaluation, CLI parsing, and
visualization. The real hierarchical checkpoint is also validated to load
strictly with all saved parameter entries.

## Research notebook and reported validation

The technical handoff notebook is located at:

```text
notebooks/chord_recognition_architecture_handoff.ipynb
```

It documents the architecture, preprocessing assumptions, validation protocol,
and limitations in more detail. The reported continuous-audio validation result
for the second-order system is 65.45% full root-quality-seventh accuracy with a
boundary F1 of 0.852. These are development/validation results, not an
untouched-test or standardized MIREX benchmark.
