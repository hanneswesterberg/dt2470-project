"""Small dependency-free helpers for reliable interactive terminal questions.

Use these functions in top-level scripts when a user should choose from known
options. Validation and retry behavior lives here so ``train.py`` and
``main.py`` can focus on their workflows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def ask_choice(question: str, options: Sequence[tuple[T, str]], default: int = 0) -> T:
    """Display numbered options and return the selected option value."""

    if not options:
        raise ValueError("At least one option is required.")
    while True:
        print(f"\n{question}")
        for index, (_, label) in enumerate(options, start=1):
            suffix = " (default)" if index - 1 == default else ""
            print(f"  {index}. {label}{suffix}")
        answer = input(f"Choose 1-{len(options)} [{default + 1}]: ").strip()
        if not answer:
            return options[default][0]
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        print("Please enter one of the displayed numbers.")


def ask_integer(question: str, default: int, minimum: int, maximum: int) -> int:
    """Ask for an integer constrained to an inclusive range."""

    while True:
        answer = input(f"{question} [{default}]: ").strip()
        if not answer:
            return default
        try:
            value = int(answer)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Please enter a number from {minimum} to {maximum}.")


def ask_text(question: str, default: str | None = None) -> str:
    """Ask for non-empty text, optionally accepting a default."""

    prompt = f"{question} [{default}]: " if default is not None else f"{question}: "
    while True:
        answer = input(prompt).strip()
        if answer:
            return answer
        if default is not None:
            return default
        print("Please enter a value.")


def ask_yes_no(question: str, default: bool = False) -> bool:
    """Ask a yes/no question and return a boolean."""

    hint = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{question} [{hint}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def ask_many(question: str, items: Sequence[tuple[T, str]]) -> list[T]:
    """Select one or more numbered items using commas, ranges, or ``all``."""

    if not items:
        raise ValueError("At least one selectable item is required.")
    while True:
        print(f"\n{question}")
        for index, (_, label) in enumerate(items, start=1):
            print(f"  {index}. {label}")
        answer = input("Choose numbers (for example 1,3-5) or 'all': ").strip().lower()
        if answer == "all":
            return [value for value, _ in items]
        try:
            selected: set[int] = set()
            for part in answer.split(","):
                if "-" in part:
                    first, last = (int(value.strip()) for value in part.split("-", 1))
                    selected.update(range(first, last + 1))
                elif part.strip():
                    selected.add(int(part.strip()))
            if selected and all(1 <= index <= len(items) for index in selected):
                return [items[index - 1][0] for index in sorted(selected)]
        except ValueError:
            pass
        print("Enter valid item numbers, ranges, or 'all'.")
