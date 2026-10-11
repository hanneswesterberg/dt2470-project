from pathlib import Path
import re

# Hardcoded Google Drive path
DRIVE_DIR = Path(
    "/Users/eliasvarling/Library/CloudStorage/"
    "GoogleDrive-elias.varling@gmail.com/My Drive/DT2470 Project"
)

DATASET_DIR = DRIVE_DIR / "full_data"

AUDIO_DIR = DATASET_DIR / "0001-1000-audio-mixes"
ANNOTATION_DIR = DATASET_DIR / "0001-1000-annotations-v1.1.0"


def find_songs():
    audio_files = {}
    annotation_files = {}

    for path in AUDIO_DIR.rglob("*_mix.flac"):
        audio_files[path.name[:4]] = path

    for path in ANNOTATION_DIR.rglob("*_beatinfo.arff"):
        annotation_files[path.name[:4]] = path

    song_ids = sorted(set(audio_files) & set(annotation_files))

    return {
        song_id: {
            "audio": audio_files[song_id],
            "annotations": annotation_files[song_id],
        }
        for song_id in song_ids
    }


if __name__ == "__main__":
    songs = find_songs()

    print(f"Found {len(songs)} songs.")

    for song_id, files in list(songs.items())[:5]:
        print(song_id, files)