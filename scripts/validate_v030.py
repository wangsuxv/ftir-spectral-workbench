#!/usr/bin/env python3
"""Run selected acceptance commands with exit codes and privacy-safe real logs."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase")
    parser.add_argument("name")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    environment = dict(os.environ)
    environment["MYPYPATH"] = str(root / "src")
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + str(root)
    environment["MPLCONFIGDIR"] = str(root / "outputs/validation-private/mpl-cache")
    started = time.monotonic()
    process = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    log = (
        "COMMAND: " + " ".join(command)
        + "\nENV: PYTHONPATH=src:. MYPYPATH=src MPLCONFIGDIR=<private cache>\n"
        + process.stdout + process.stderr
        + f"\nEXIT_CODE: {process.returncode}\nELAPSED_SECONDS: {elapsed:.3f}\n"
    )
    private = root / "outputs/validation-private" / args.phase
    private.mkdir(parents=True, exist_ok=True)
    (private / f"{args.name}.log").write_text(log, encoding="utf-8")
    public = root / "artifacts/validation/v0.3.0" / args.phase
    public.mkdir(parents=True, exist_ok=True)
    log = log.replace(str(root), "<repository>").replace(str(Path.home()), "<home>")
    (public / f"{args.name}.log").write_text(
        "# Real command output; local repository/home paths redacted.\n" + log,
        encoding="utf-8",
    )
    print(log[-5000:])
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
