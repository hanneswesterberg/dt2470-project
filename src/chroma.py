
import numpy as np
import librosa

SR = 22050
HOP_LENGTH = 512
N_BINS = 84


def compute_cqt(audio_path):
    """
Compute the magnitude Constant-Q Transform (CQT) of an audio file.

The CQT converts an audio signal into a time-frequency representation
where frequency bins are logarithmically spaced to match musical notes.

Unlike a standard Fourier Transform, which uses equally spaced
frequencies (Hz), the CQT uses 12 bins per octave, corresponding
to the 12 semitones in Western music (C, C#, D, ..., B).

We compute 84 frequency bins starting at C1, covering 7 octaves
(C1 to B7). Each bin represents the magnitude of a musical pitch
at a particular moment in time.

This allows us to analyze which musical notes are present in the
audio before combining them into chroma vectors.

Returns:
    C: CQT magnitude matrix of shape (84, n_frames).
       Rows represent musical pitches (semitones).
       Columns represent time frames.
       Values represent spectral magnitude.

    duration: Total audio duration in seconds.
"""
    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    C = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz("C1"),
            n_bins=N_BINS,
            bins_per_octave=12,
        )
    )

    return C, len(y) / sr


def compute_chroma(C, start, end, low_bin=0, high_bin=84):
    """
    Compute a 12-dimensional chroma vector for any time interval
    and any frequency range within the CQT.

    Parameters:
        C: CQT magnitude matrix from compute_cqt().
        start, end: Time interval in seconds.
        low_bin, high_bin: CQT frequency-bin boundaries.
                           high_bin is exclusive.

    Returns:
        chroma: Normalized vector of shape (12,).
    """
    if end <= start:
        raise ValueError("End time must be greater than start time.")

    if not (0 <= low_bin < high_bin <= C.shape[0]):
        raise ValueError("Invalid CQT frequency-bin range.")

    # Ignore the outer 5% at each end of the interval
    duration = end - start
    start += 0.05 * duration
    end -= 0.05 * duration

    start_frame = int(start * SR / HOP_LENGTH)
    end_frame = int(end * SR / HOP_LENGTH)

    start_frame = max(0, min(start_frame, C.shape[1] - 1))
    end_frame = max(start_frame + 1, min(end_frame, C.shape[1]))

    # Average spectral magnitudes over the selected interval
    avg = C[low_bin:high_bin, start_frame:end_frame].mean(axis=1)

    # Fold octaves into 12 pitch classes (C, C#, ..., B)
    chroma = np.zeros(12)

    for i, value in enumerate(avg):
        chroma[(low_bin + i) % 12] += value

    # Normalize
    chroma /= max(np.linalg.norm(chroma), 1e-10)

    return chroma


def compute_chroma_24d(C, start, end):
    """
    Compute our model's 24D chroma feature vector.

    Bass:     C1–B3 -> CQT bins 0:36
    Midrange: C4–B5 -> CQT bins 36:60

    Returns:
        x: Array of shape (24,).
    """
    bass = compute_chroma(C, start, end, low_bin=0, high_bin=36)
    mid = compute_chroma(C, start, end, low_bin=36, high_bin=60)

    return np.concatenate([bass, mid])


def compute_chroma_intervals(C, intervals, feature_fn=compute_chroma_24d):
    """
    Extract features for multiple arbitrary time intervals.

    Parameters:
        C: CQT magnitude matrix.
        intervals: Array of shape (n_intervals, 2).
        feature_fn: Function to use for each interval.

    Returns:
        X: Feature matrix (n_intervals, n_features).
    """
    return np.asarray([
        feature_fn(C, start, end)
        for start, end in intervals
    ])


if __name__ == "__main__":
    from data.load_data import find_songs

    songs = find_songs()
    song_id = next(iter(songs))

    C, duration = compute_cqt(songs[song_id]["audio"])

    # Example interval
    start, end = 5.0, 6.0

    # Standard 12D chroma using all seven octaves
    chroma_12d = compute_chroma(C, start, end)

    # Our 24D representation (bass + midrange)
    chroma_24d = compute_chroma_24d(C, start, end)

    print("Song:", song_id)
    print("Audio duration:", round(duration, 2), "seconds")
    print("12D chroma shape:", chroma_12d.shape)
    print("24D chroma shape:", chroma_24d.shape)
    print("24D feature vector:", chroma_24d)
