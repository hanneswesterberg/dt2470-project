"""Create and update the project's two isolated Python environments.

The application environment uses the user's current Python and
``requirements.txt``. BeatNet uses a separate uv-managed Python 3.9 environment
because madmom and its numerical dependencies are incompatible with the modern
application stack.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ENV = PROJECT_ROOT / ".venv"
BEATNET_ENV = PROJECT_ROOT / ".venv-beatnet"


def environment_python(environment: Path) -> Path:
    """Return the platform-specific Python executable inside a virtual environment."""

    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return environment / relative


def _requirements_hash(path: Path, extra: str = "") -> str:
    digest = hashlib.sha256(path.read_bytes())
    digest.update(extra.encode("utf-8"))
    return digest.hexdigest()


def _is_current_environment(environment: Path) -> bool:
    try:
        return Path(sys.prefix).resolve() == environment.resolve()
    except OSError:
        return False


def ensure_application_environment() -> Path:
    """Create/update ``.venv`` and return its Python executable."""

    requirements = PROJECT_ROOT / "requirements.txt"
    marker = APP_ENV / ".requirements.sha256"
    wanted = _requirements_hash(requirements)
    python = environment_python(APP_ENV)
    if not python.is_file():
        print("Creating application environment in .venv ...")
        subprocess.run([sys.executable, "-m", "venv", str(APP_ENV)], check=True)
    installed = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    if installed != wanted:
        print("Installing application dependencies ...")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(requirements)], check=True
        )
        marker.write_text(wanted, encoding="utf-8")
    return python


def restart_in_application_environment(script: Path) -> None:
    """Restart a top-level script inside ``.venv`` when necessary."""

    if _is_current_environment(APP_ENV):
        return
    python = ensure_application_environment()
    os.execv(str(python), [str(python), str(script), *sys.argv[1:]])


def _ensure_portaudio() -> None:
    """Install PortAudio through Homebrew on macOS when it is missing."""

    if sys.platform != "darwin":
        return
    brew = shutil.which("brew")
    if brew is None:
        raise RuntimeError(
            "Homebrew is required to install PortAudio for BeatNet/PyAudio."
        )
    result = subprocess.run(
        [brew, "list", "--versions", "portaudio"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        print("Installing PortAudio with Homebrew ...")
        subprocess.run([brew, "install", "portaudio"], check=True)


def _ensure_uv(app_python: Path) -> None:
    result = subprocess.run(
        [str(app_python), "-m", "uv", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Installing uv environment manager ...")
        subprocess.run([str(app_python), "-m", "pip", "install", "uv"], check=True)


def ensure_beatnet_environment() -> Path:
    """Create/update the isolated Python 3.9 BeatNet environment."""

    app_python = ensure_application_environment()
    _ensure_uv(app_python)
    _ensure_portaudio()
    requirements = PROJECT_ROOT / "requirements-beatnet.txt"
    marker = BEATNET_ENV / ".requirements.sha256"
    wanted = _requirements_hash(requirements, extra="madmom==0.16.1\nBeatNet")
    python = environment_python(BEATNET_ENV)
    installed = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    if python.is_file() and installed == wanted:
        return python

    print("Preparing isolated Python 3.9 BeatNet environment ...")
    subprocess.run(
        [str(app_python), "-m", "uv", "python", "install", "3.9"], check=True
    )
    if not python.is_file():
        subprocess.run(
            [
                str(app_python),
                "-m",
                "uv",
                "venv",
                str(BEATNET_ENV),
                "--python",
                "3.9",
                "--seed",
            ],
            check=True,
        )
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel"], check=True
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(requirements)], check=True
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-cache-dir",
            "madmom==0.16.1",
        ],
        check=True,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "BeatNet"], check=True
    )
    subprocess.run(
        [str(python), "-c", "from BeatNet.BeatNet import BeatNet"], check=True
    )
    marker.write_text(wanted, encoding="utf-8")
    return python


def beatnet_environment_available() -> bool:
    """Return whether a verified BeatNet environment is already installed."""

    python = environment_python(BEATNET_ENV)
    requirements = PROJECT_ROOT / "requirements-beatnet.txt"
    marker = BEATNET_ENV / ".requirements.sha256"
    if not python.is_file() or not marker.is_file():
        return False
    wanted = _requirements_hash(requirements, extra="madmom==0.16.1\nBeatNet")
    return marker.read_text(encoding="utf-8").strip() == wanted
