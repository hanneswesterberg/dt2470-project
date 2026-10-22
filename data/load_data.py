"""Locate paired AAM audio and annotation files.

The full AAM corpus stays in Google Drive. This module only returns paths into
the mounted Drive folder; it never copies audio into the repository.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

AUDIO_SUFFIXES = {".flac", ".mp3", ".ogg", ".wav"}


@dataclass(frozen=True)
class SongFiles:
    """The audio and annotation belonging to one song."""

    song_id: str
    audio: Path
    annotations: Path


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read the small ``KEY=VALUE`` subset used by this project."""

    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def default_drive_dir(env_file: Path | str = ".env") -> Path:
    """Find the mounted Google Drive project folder.

    Explicit ``AAM_DRIVE_DIR`` wins. Otherwise the configured Google account is
    used; with one matching Drive mount, the account can be detected
    automatically. The shared folder name defaults to ``DT2470 Project``.
    """

    file_values = _read_dotenv(Path(env_file))
    direct = os.environ.get("AAM_DRIVE_DIR", file_values.get("AAM_DRIVE_DIR"))
    if direct:
        return Path(direct).expanduser()
    email = os.environ.get("GOOGLE_DRIVE_EMAIL", file_values.get("GOOGLE_DRIVE_EMAIL"))
    project = os.environ.get(
        "AAM_PROJECT_FOLDER", file_values.get("AAM_PROJECT_FOLDER", "DT2470 Project")
    )
    cloud_storage = Path.home() / "Library" / "CloudStorage"
    if email:
        drive_root = cloud_storage / f"GoogleDrive-{email}"
        shared_candidates = sorted(drive_root.glob(f".shortcut-targets-by-id/*/{project}"))
        candidates = [
            *shared_candidates,
            drive_root / "My Drive" / project,
        ]
    else:
        candidates = sorted(
            path
            for drive_root in cloud_storage.glob("GoogleDrive-*")
            for path in (
                drive_root / "My Drive" / project,
                *drive_root.glob(f".shortcut-targets-by-id/*/{project}"),
            )
        )
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    if len(existing) == 1:
        return existing[0]
    if not existing:
        raise RuntimeError(
            "Could not locate the shared Google Drive project folder. Set "
            "AAM_DRIVE_DIR, or GOOGLE_DRIVE_EMAIL and AAM_PROJECT_FOLDER, in .env."
        )
    if len(existing) > 1 and email:
        shared = [path for path in existing if ".shortcut-targets-by-id" in path.parts]
        if len(shared) == 1:
            return shared[0]
    if len(existing) > 1:
        raise RuntimeError(
            "Found multiple Google Drive project folders. Set AAM_DRIVE_DIR "
            "to the folder used for this project."
        )
    return existing[0]


def _song_id(path: Path) -> str:
    """Return a stable ID for common AAM file names."""

    name = path.stem
    for suffix in ("_beatinfo", "_mix"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _index_files(paths: Iterable[Path]) -> dict[str, Path]:
    return {_song_id(path): path for path in sorted(paths)}


def find_songs(
    audio_dir: Path | str | None = None,
    annotation_dir: Path | str | None = None,
) -> dict[str, SongFiles]:
    """Find songs for which both audio and beat annotations are available.

    Explicit parameters may point to arbitrary directories. If omitted, they
    resolve to the 0001--1000 folders in the mounted Google Drive.
    """

    drive = None
    if audio_dir is None or annotation_dir is None:
        drive = default_drive_dir()
    audio_root = (
        Path(audio_dir)
        if audio_dir is not None
        else drive / "full_data" / "0001-1000-audio-mixes"
    )
    annotation_root = (
        Path(annotation_dir)
        if annotation_dir is not None
        else drive / "full_data" / "0001-1000-annotations-v1.1.0"
    )
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Audio dataset folder does not exist: {audio_root}")
    if not annotation_root.is_dir():
        raise FileNotFoundError(
            f"Annotation dataset folder does not exist: {annotation_root}"
        )

    audio = _index_files(
        path
        for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    annotations = _index_files(annotation_root.rglob("*_beatinfo.arff"))
    return {
        song_id: SongFiles(song_id, audio[song_id], annotations[song_id])
        for song_id in sorted(audio.keys() & annotations.keys())
    }


def find_input_audio(input_dir: Path | str) -> list[Path]:
    """Return supported audio files directly inside the input folder."""

    root = Path(input_dir)
    root.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def find_input_annotation(audio_path: Path) -> Path | None:
    """Find an optional AAM annotation next to an input audio file."""

    base = _song_id(audio_path)
    candidates = (
        audio_path.with_name(f"{base}_beatinfo.arff"),
        audio_path.with_name(f"{base}.arff"),
        audio_path.parent / "annotations" / f"{base}_beatinfo.arff",
    )
    return next((path for path in candidates if path.is_file()), None)
