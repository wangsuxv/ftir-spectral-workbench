# Phase 0 — actual starting point (2026-09-10)

- Repository subdirectory: `ftir-spectral-workbench` (the task working directory is its parent).
- Initial HEAD: `8cf6cc8bd38491d6549595564f2c57715bc69ef0`.
- Initial branch: `feat/v0.2.5-post-baseline-smoothing`; tracked worktree clean.
- New local branch: `feat/v0.3-independent-batch-baseline`.
- Specification reference `c0751bc2fa0e3a3716c70bc9da33766fcad2b746` is not in the local Git object database. No reset, fetch, push or history replacement was performed.
- Runtime: existing `.venv`, Python 3.12.14; numerical package versions match the applicable `requirements.lock` entries (see `environment.json`). No dependencies were upgraded.
- Actual baseline pytest: **723 passed, 5 warnings**, 65.24 seconds. These are this run's results, not copied historical evidence.
- Ruff: passed.
- Requested unqualified `python -m mypy`: exit 2, editable-package discovery reports missing `py.typed`. Running with `MYPYPATH=src` resolves source discovery: **20 source files passed**. Both outputs are retained. Runtime module imports resolve to this checkout.
- Existing v0.2.1 freeze audit: **34/34**, no added/missing/changed frozen files. Manifest itself is unchanged.

Logs preserve actual output, with local repository/home paths replaced by placeholders. Unredacted runtime logs remain in ignored `outputs/validation-private/`. All new acceptance data is synthetic; no experimental data was opened.
