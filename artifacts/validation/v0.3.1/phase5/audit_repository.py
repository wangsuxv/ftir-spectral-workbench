"""Read-only final repository evidence; inspect changed public files only."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

BASE = "9fc13c10ae34bd53cb220d0807edb18c9e2e41dc"
ROOT = Path.cwd()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


tracked_changes = set(git("diff", "--name-only", BASE).decode().splitlines())
untracked = set(git("ls-files", "--others", "--exclude-standard").decode().splitlines())
changed = sorted(tracked_changes | untracked)
manifest = "artifacts/v0.2.1_science_freeze_manifest.json"
assert (ROOT / manifest).read_bytes() == git("show", f"{BASE}:{manifest}")
assert (ROOT / "requirements.lock").read_bytes() == git("show", f"{BASE}:requirements.lock")
old_smoothing = git("show", f"{BASE}:src/ftir_workbench/post_baseline_smoothing.py")
assert (ROOT / "src/ftir_workbench/post_baseline_smoothing.py").read_bytes().startswith(old_smoothing)
for path in (
    "src/ftir_workbench/batch/export.py", "src/ftir_workbench/batch/service.py",
    "src/ftir_workbench/batch/state.py", "src/ftir_workbench/batch/importing.py",
    "ui/streamlit_app.py",
):
    assert (ROOT / path).read_bytes() == git("show", f"{BASE}:{path}"), path
private_paths = []
unexpected_data = []
# The literal patterns themselves are not machine paths.
for name in changed:
    path = ROOT / name
    assert not name.startswith(("outputs/", ".venv/", "data/original/"))
    if path.is_file() and path.suffix.lower() in {".csv", ".zip", ".npy", ".npz", ".xlsx", ".bin"}:
        unexpected_data.append(name)
    if path.is_file() and path.suffix in {".log", ".json", ".md"}:
        content = path.read_text(encoding="utf-8")
        if re.search(r"/Users/[A-Za-z0-9_.-]+/|/private/var/folders/[a-z0-9]{2}/|/var/folders/[a-z0-9]{2}/", content):
            private_paths.append(name)
assert not private_paths, private_paths
assert not unexpected_data, unexpected_data
git("diff", "--check")
report = {
    "status": "pass",
    "baseline_commit": BASE,
    "audited_head": git("rev-parse", "HEAD").decode().strip(),
    "branch": git("branch", "--show-current").decode().strip(),
    "manifest_exact_baseline_bytes": True,
    "requirements_lock_exact_baseline_bytes": True,
    "old_smoothing_implementation_prefix_bytes": len(old_smoothing),
    "old_smoothing_prefix_sha256": hashlib.sha256(old_smoothing).hexdigest(),
    "old_smoothing_prefix_unchanged": True,
    "old_batch_import_baseline_state_export_and_unified_entry_unchanged": True,
    "changed_public_paths": changed,
    "changed_public_path_count": len(changed),
    "private_machine_path_leaks": private_paths,
    "new_spectrum_binary_or_csv_files": unexpected_data,
    "scope": "Only changed/untracked nonignored public files were read. Ignored spectra and private downloads were not read.",
}
output = ROOT / "artifacts/validation/v0.3.1/phase5/repository_audit.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
