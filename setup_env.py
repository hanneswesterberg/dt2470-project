"""Explicit one-command setup for the application and optional BeatNet environments."""

from __future__ import annotations

import argparse

from scripts.environments import (
    ensure_application_environment,
    ensure_beatnet_environment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare project virtual environments."
    )
    parser.add_argument(
        "--with-beatnet",
        action="store_true",
        help="Also install the isolated Python 3.9 BeatNet environment.",
    )
    args = parser.parse_args()
    app_python = ensure_application_environment()
    print(f"Application environment ready: {app_python}")
    if args.with_beatnet:
        beatnet_python = ensure_beatnet_environment()
        print(f"BeatNet environment ready: {beatnet_python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
