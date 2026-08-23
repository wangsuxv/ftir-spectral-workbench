"""Deterministic SHA-256 primitives for workbench dependency tracking."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:  # pragma: no cover
    from ftir_baseline.pipeline import PipelineResult

    from .config import TwoDCOSConfig, WorkbenchProjectConfig
    from .models import PreparedSpectralDataset


def _canonical_value(value: Any, *, path: str = "root") -> Any:
    """Convert supported scientific/config objects to canonical JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _canonical_value(value.item(), path=path)
    if isinstance(value, Enum):
        return _canonical_value(value.value, path=path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            raise TypeError(f"{path} contains a complex array")
        return _canonical_value(value.tolist(), path=path)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key in output:
                raise ValueError(f"{path} contains mapping keys that collapse to {key!r}")
            output[key] = _canonical_value(item, path=f"{path}.{key}")
        return output
    if isinstance(value, (tuple, list)):
        return [
            _canonical_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        converted = [_canonical_value(item, path=f"{path}[]") for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_value(
                getattr(value, item.name),
                path=f"{path}.{item.name}",
            )
            for item in fields(value)
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _canonical_value(
            model_dump(mode="json", by_alias=True),
            path=path,
        )
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical_value(to_dict(), path=path)
    raise TypeError(
        f"{path} contains unsupported value of type {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using the project's canonical UTF-8 JSON form."""

    normalized = _canonical_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _length_prefix(payload: bytes) -> bytes:
    return len(payload).to_bytes(8, byteorder="big", signed=False) + payload


def _float64_little_endian(values: ArrayLike, *, field_name: str) -> np.ndarray:
    if np.iscomplexobj(values):
        raise TypeError(f"{field_name} must contain real numeric values")
    try:
        array = np.asarray(values, dtype=np.float64, order="C")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must contain real numeric values") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} contains NaN or infinite values")
    # ``astype`` also handles native big-endian platforms deterministically.
    return np.ascontiguousarray(array.astype(np.dtype("<f8"), copy=False))


def update_array_hash(
    digest: Any,
    values: ArrayLike,
    *,
    field_name: str,
) -> None:
    """Update a SHA-256-like digest with field name, shape, dtype, and bytes."""

    if not isinstance(field_name, str) or not field_name:
        raise ValueError("field_name must be a non-empty string")
    array = _float64_little_endian(values, field_name=field_name)
    header = canonical_json_bytes(
        {
            "field_name": field_name,
            "shape": list(array.shape),
            "dtype": "<f8",
            "order": "C",
        }
    )
    payload = array.tobytes(order="C")
    digest.update(_length_prefix(header))
    digest.update(_length_prefix(payload))


def array_sha256(values: ArrayLike, *, field_name: str = "array") -> str:
    """Hash a finite array in canonical little-endian ``float64`` form."""

    digest = hashlib.sha256()
    update_array_hash(digest, values, field_name=field_name)
    return digest.hexdigest()


def prepared_data_sha256(
    wavenumber: ArrayLike,
    perturbation: ArrayLike,
    perturbation_labels: tuple[str, ...] | list[str],
    spectra: ArrayLike,
    *,
    normalization_state: str = "none",
) -> str:
    """Hash every scientific input that enters prepared-only 2D-COS.

    ``none`` and ``display_only`` deliberately have the same dependency marker:
    display normalization does not change the spectra or invalidate matrices.
    An explicit scientific-normalization branch receives a distinct marker even
    in the unusual case where its transformed values equal the main data.
    """

    if normalization_state not in {"none", "display_only", "scientific_explicit"}:
        raise ValueError(f"unsupported normalization_state: {normalization_state!r}")
    labels = tuple(perturbation_labels)
    if any(not isinstance(label, str) for label in labels):
        raise TypeError("perturbation_labels must contain strings")
    branch_kind = (
        "scientific_explicit"
        if normalization_state == "scientific_explicit"
        else "primary_analysis"
    )
    digest = hashlib.sha256()
    digest.update(
        _length_prefix(
            canonical_json_bytes(
                {
                    "schema": "ftir-workbench-prepared-data-v1",
                    "perturbation_labels": labels,
                    "branch_kind": branch_kind,
                }
            )
        )
    )
    update_array_hash(digest, wavenumber, field_name="wavenumber")
    update_array_hash(digest, perturbation, field_name="perturbation")
    update_array_hash(digest, spectra, field_name="spectra")
    return digest.hexdigest()


def _baseline_scientific_config(config: Any) -> dict[str, Any]:
    payload = _canonical_value(config, path="baseline_config")
    if not isinstance(payload, dict):
        raise TypeError("baseline config must serialize to a mapping")
    # Normalization is a separate non-destructive branch in ftir_baseline.
    # Main prepared data always uses analysis_data, so it is not a dependency.
    payload.pop("normalization", None)
    return payload


def baseline_fingerprint(
    result: PipelineResult,
    *,
    scientific_branch: Mapping[str, Any] | None = None,
    spectra: ArrayLike | None = None,
) -> str:
    """Fingerprint baseline scientific inputs, config, and corrected absorbance.

    For the primary path, ``result.analysis_data`` is always selected internally.
    A caller creating an explicit scientific-normalization sensitivity branch must
    supply both a branch descriptor and its transformed spectra.
    """

    if (scientific_branch is None) != (spectra is None):
        raise ValueError(
            "scientific_branch and spectra must be supplied together for an explicit branch"
        )
    selected = result.analysis_data if spectra is None else spectra
    digest = hashlib.sha256()
    payload = {
        "schema": "ftir-workbench-baseline-v1",
        "source_sha256": str(result.input_sha256),
        "config": _baseline_scientific_config(result.config),
        "perturbation_labels": tuple(result.absorbance_selected.perturbation_labels),
        "scientific_branch": None
        if scientific_branch is None
        else _canonical_value(scientific_branch, path="scientific_branch"),
    }
    digest.update(_length_prefix(canonical_json_bytes(payload)))
    update_array_hash(
        digest,
        result.absorbance_selected.wavenumber,
        field_name="wavenumber",
    )
    update_array_hash(
        digest,
        result.absorbance_selected.perturbation,
        field_name="perturbation",
    )
    update_array_hash(digest, selected, field_name="corrected_absorbance")
    return digest.hexdigest()


def twodcos_fingerprint(
    prepared: PreparedSpectralDataset | str,
    config: TwoDCOSConfig,
) -> str:
    """Fingerprint prepared data and scientific 2D settings, excluding display."""

    parent_hash = (
        prepared.prepared_data_sha256
        if hasattr(prepared, "prepared_data_sha256")
        else str(prepared)
    )
    scientific_dict = getattr(config, "scientific_dict", None)
    config_payload = scientific_dict() if callable(scientific_dict) else config
    return canonical_json_sha256(
        {
            "schema": "ftir-workbench-2dcos-v1",
            "prepared_data_sha256": parent_hash,
            "config": config_payload,
        }
    )


def project_fingerprint(
    config: WorkbenchProjectConfig,
    *,
    prepared: PreparedSpectralDataset | str | None = None,
) -> str:
    """Fingerprint the project's scientific dependency graph.

    Display-only settings are intentionally excluded.  The optional prepared
    hash links an instantiated project to its current baseline output.
    """

    scientific_dict = getattr(config, "scientific_dict", None)
    config_payload = scientific_dict() if callable(scientific_dict) else config
    prepared_hash: str | None
    if prepared is None:
        prepared_hash = None
    elif hasattr(prepared, "prepared_data_sha256"):
        prepared_hash = prepared.prepared_data_sha256
    else:
        prepared_hash = str(prepared)
    return canonical_json_sha256(
        {
            "schema": "ftir-workbench-project-v1",
            "config": config_payload,
            "prepared_data_sha256": prepared_hash,
        }
    )


# Readable aliases for callers that prefer verb-first naming.
fingerprint_array = array_sha256
fingerprint_baseline = baseline_fingerprint
fingerprint_project = project_fingerprint
fingerprint_twodcos = twodcos_fingerprint


__all__ = [
    "array_sha256",
    "baseline_fingerprint",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "fingerprint_array",
    "fingerprint_baseline",
    "fingerprint_project",
    "fingerprint_twodcos",
    "prepared_data_sha256",
    "project_fingerprint",
    "twodcos_fingerprint",
    "update_array_hash",
]
