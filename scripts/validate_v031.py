#!/usr/bin/env python3
"""Run v0.3.1 acceptance commands, retaining raw and privacy-safe real logs."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
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
    environment["MPLCONFIGDIR"] = str(root / "outputs/validation-private/v031/mpl-cache")
    started_utc = datetime.now(UTC).isoformat()
    started = time.monotonic()
    process = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    log = (
        "COMMAND: " + shlex.join(command)
        + "\nSTARTED_UTC: " + started_utc
        + "\nENV: PYTHONPATH=src:. MYPYPATH=src MPLCONFIGDIR=<private v031 cache>\n"
        + process.stdout + process.stderr
        + f"\nEXIT_CODE: {process.returncode}\nELAPSED_SECONDS: {elapsed:.3f}\n"
    )
    private = root / "outputs/validation-private/v031" / args.phase
    public = root / "artifacts/validation/v0.3.1" / args.phase
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    stem = args.name
    counter = 1
    while (private / f"{stem}.log").exists() or (public / f"{stem}.log").exists():
        counter += 1
        stem = f"{args.name}.{counter:03d}"
    (private / f"{stem}.log").write_text(log, encoding="utf-8")
    log = log.replace(str(root), "<repository>").replace(str(Path.home()), "<home>")
    (public / f"{stem}.log").write_text(
        "# Real command output; local repository/home paths redacted.\n" + log,
        encoding="utf-8",
    )
    print(log[-6000:])
    print(f"LOG: artifacts/validation/v0.3.1/{args.phase}/{stem}.log")
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
