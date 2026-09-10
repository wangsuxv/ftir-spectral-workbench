"""Strict public-parser adaptation: each file is read, then each column separated."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import numpy as np

from ftir_baseline.config import IntensityUnit
from ftir_baseline.io import TextImportOptions, probe_spectrum_file, read_spectrum_file

from .models import (
    BatchWorkspace,
    ImportedSource,
    PreparationConfig,
    SpectrumProcessingState,
    SpectrumRecord,
)


def import_sources(
    workspace: BatchWorkspace,
    uploads: Sequence[tuple[str, bytes]],
    *,
    input_unit: IntensityUnit = "absorbance",
    options: TextImportOptions | None = None,
) -> list[str]:
    """Import explicitly confirmed source units; failures are isolated by file.

    Call once per source when source defaults differ. A stable source ID scopes
    every original filename, including identical names. No acquisition ordering,
    coordinate alignment, interpolation, or implicit duplicate deletion occurs.
    """

    selected_options = options or TextImportOptions()
    created: list[str] = []
    for original_name, contents in uploads:
        source_id = uuid4().hex
        file_hash = hashlib.sha256(contents).hexdigest()
        try:
            # Only preserve a suffix in the temporary name; uploaded paths are never used.
            suffix = Path(original_name.replace("\\", "/")).suffix
            with tempfile.TemporaryDirectory(prefix="ftir-independent-") as directory:
                path = Path(directory) / f"source{suffix}"
                path.write_bytes(contents)
                probe = probe_spectrum_file(path, options=selected_options)
                loaded = read_spectrum_file(
                    path,
                    input_unit=input_unit,
                    perturbation=[float(index) for index in range(probe.columns - 1)],
                    source_name=original_name,
                    sort_by_perturbation=False,
                    import_options=selected_options,
                )
            evidence = probe.to_dict()
            evidence["source_name"] = original_name
            source = ImportedSource(
                source_id=source_id,
                original_filename=original_name,
                original_bytes_sha256=file_hash,
                original_bytes=contents,
                default_input_unit=input_unit,
                import_options=selected_options.to_dict(),
                import_probe=evidence,
            )
            duplicate = any(
                item.original_bytes_sha256 == file_hash for item in workspace.sources.values()
            )
            records: list[SpectrumRecord] = []
            for index in range(loaded.n_spectra):
                label = (
                    probe.header[index + 1]
                    if probe.header is not None
                    else (
                        Path(original_name).stem
                        if loaded.n_spectra == 1
                        else f"spectrum_{index}"
                    )
                )
                records.append(SpectrumRecord(
                    spectrum_id=uuid4().hex,
                    source_id=source_id,
                    original_column_index=index + 1,
                    original_column_label=label,
                    display_name=label,
                    wavenumber=loaded.wavenumber,
                    raw_intensity=loaded.spectra[index],
                    confirmed_input_unit=input_unit,
                    original_axis_direction=loaded.axis_direction,
                    duplicate_candidate=duplicate,
                ))
            workspace.sources[source_id] = source
            for record in records:
                preparation = PreparationConfig(
                    input_unit=input_unit,
                    wavenumber_range=(
                        float(np.max(record.wavenumber)), float(np.min(record.wavenumber))
                    ),
                )
                workspace.records[record.spectrum_id] = record
                workspace.states[record.spectrum_id] = SpectrumProcessingState(
                    preparation_draft=preparation,
                    preparation_committed=PreparationConfig(**preparation.to_dict()),
                )
                workspace.display_order.append(record.spectrum_id)
                created.append(record.spectrum_id)
        except (ValueError, TypeError, OSError) as exc:
            # Parser location evidence stays intact while ephemeral local paths do not leak.
            detail = str(exc)
            if "path" in locals():
                detail = detail.replace(str(path), original_name).replace(path.name, original_name)
            workspace.import_issues.append({
                "code": "IMPORT_FAILED", "source_id": source_id,
                "original_filename": original_name, "source_sha256": file_hash,
                "message": detail,
            })
    if workspace.selected_spectrum_id is None and created:
        workspace.selected_spectrum_id = created[0]
    if created:
        workspace.last_export_summary = None
    return created
