# Architecture and musical interpretation

## What the system predicts

The main model describes each chord with four parts:

1. **Root** – the pitch name, such as C, F#, or Bb.
2. **Quality** – the basic chord type, mainly major, minor, dominant, diminished,
   or suspended forms in the training vocabulary.
3. **Seventh** – whether the chord includes a minor seventh, major seventh,
   diminished seventh, or no seventh.
4. **Extension** – added tones such as 9ths, 11ths, and 13ths when they are
   present in the labels.

This is a useful musical decomposition. The root tells us where the harmony is
centred, while quality and seventh describe the colour and function of the chord.
Extensions add extra colour but are harder to learn because they are often quiet
in a recording and are less consistently annotated.

## Hierarchical CNN

The input is a constant-Q time-frequency representation. It keeps the harmonic
relationships between notes visible: shifting a chord up by a semitone shifts
its spectral pattern in a predictable way.

The CNN first learns shared acoustic features and then uses separate output heads
for root, quality, seventh, and extension. The later heads receive the earlier
predictions as context. This matches the musical structure of the problem: it is
easier to decide whether a sound is a dominant seventh after estimating its root
and basic quality.

The heads are trained with multitask cross-entropy losses. During inference the
model produces probabilities for each part, which are combined into chord-state
emissions for the temporal decoder.

## Second-order HSMM

A frame-by-frame classifier can change chord onsets too often. The decoder fixes
this by looking at a rolling window of emissions and remembering the previous two
chords. It is a second-order hidden semi-Markov model (HSMM):

- the emission probability measures how well the audio matches a chord;
- transition probabilities describe common harmonic movement;
- duration probabilities discourage unrealistically short chord segments;
- the second-order history captures patterns such as tonic–predominant–dominant
  motion and repeated progressions.

Viterbi decoding finds the most likely sequence of chord states over the whole
song. The result is therefore musically smoother than choosing the most likely
CNN output independently at every frame.

## Baseline

The original Gaussian/HMM pipeline is kept as an optional baseline. It uses the
older 24-state major/minor representation and can be selected with
`--pipeline gaussian`. The hierarchical CNN + HSMM is the default because it can
represent more chord detail while keeping the temporal modelling explicit.

## Evaluation

`evaluate_rwc.py` evaluates the RWC feature files against their annotations.
`main.py --source rwc` runs the complete audio pipeline when the RWC audio is
available through Drive. The viewer shows the prediction and, when a matching
annotation CSV exists, the ground-truth chord timeline as well.
