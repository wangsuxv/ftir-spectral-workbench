"""Dependency hashes separate numeric input, source selection, and stage identity."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any

from ftir_baseline.config import PipelineConfig
from ftir_baseline.models import thaw_mapping

from .models import BatchWorkspace, Stage, scientific_input_hash


def json_fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def implementation_fingerprint() -> str:
    """Bind actual scientific source bytes and installed numerical dependencies.

    The value is session-local and stable; reload the process after updating code.
    Paths are used only to read files and never written into the fingerprint payload.
    """

    source_root = Path(__file__).resolve().parents[2]
    files = sorted((source_root / "ftir_baseline").rglob("*.py"))
    files.extend(Path(__file__).parent / name for name in (
        "models.py", "importing.py", "recipes.py", "fingerprints.py", "service.py", "state.py"
    ))
    payload = {
        "files": {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in files},
        "versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "pybaselines", "pydantic"
        )},
        "python": platform.python_version(),
        "adapter_contract": "independent-batch-v1",
    }
    return json_fingerprint(payload)


def source_selection_fingerprint(workspace: BatchWorkspace, spectrum_id: str) -> str:
    record = workspace.records[spectrum_id]
    source = workspace.sources[record.source_id]
    return json_fingerprint({
        "source_sha256": source.original_bytes_sha256,
        "column_index": record.original_column_index,
        "import_options": thaw_mapping(source.import_options),
    })


def stage_fingerprint(
    workspace: BatchWorkspace,
    spectrum_id: str,
    config: PipelineConfig,
    stage: Stage,
    parent_coarse_fingerprint: str | None = None,
    *,
    implementation: str | None = None,
) -> str:
    record = workspace.records[spectrum_id]
    source = workspace.sources[record.source_id]
    return json_fingerprint({
        "workspace_id": workspace.workspace_id,
        "spectrum_id": spectrum_id,
        "source_id": record.source_id,
        "original_filename": source.original_filename,
        "original_column_label": record.original_column_label,
        "source_selection_sha256": source_selection_fingerprint(workspace, spectrum_id),
        "scientific_input_sha256": scientific_input_hash(
            record.wavenumber, record.raw_intensity, config.input_unit
        ),
        "workflow_mode": "independent_batch",
        "stage": stage,
        "config": config.to_dict(),
        "parent_coarse_fingerprint": parent_coarse_fingerprint,
        "implementation": implementation or implementation_fingerprint(),
    })
