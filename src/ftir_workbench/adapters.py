"""In-memory adapters between the authoritative baseline and 2D packages."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike

from ftir_baseline.pipeline import PipelineResult

from .fingerprints import baseline_fingerprint, prepared_data_sha256
from .models import AxisDirection, NormalizationState, PreparedSpectralDataset
from .validation import wavenumber_direction


def _baseline_run_id(value: str | None) -> str:
    if value is None:
        return f"baseline-{uuid4().hex}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("baseline_run_id must be a non-empty string")
    return value.strip()


def _original_axis_direction(result: PipelineResult) -> AxisDirection:
    metadata = result.absorbance_selected.metadata
    recorded = metadata.get(
        "original_axis_direction",
        result.raw_input.metadata.get("original_axis_direction"),
    )
    if recorded in {"ascending", "descending"}:
        return cast(AxisDirection, recorded)
    return wavenumber_direction(result.raw_input.wavenumber)


def _perturbation_policy(result: PipelineResult) -> str:
    metadata = result.absorbance_selected.metadata
    explicit = metadata.get("perturbation_order_policy") or metadata.get("order_policy")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if bool(metadata.get("sorted_by_perturbation")):
        return "sort_by_perturbation"
    return "preserve_file_order"


def _normalization_state_for_primary(result: PipelineResult) -> NormalizationState:
    return "display_only" if result.normalization.method == "minmax_display" else "none"


def _qc_dict(result: PipelineResult) -> dict[str, Any]:
    as_dict = getattr(result.qc, "as_dict", None)
    if callable(as_dict):
        value = as_dict()
        return dict(value) if isinstance(value, Mapping) else {"value": value}
    if isinstance(result.qc, Mapping):
        return deepcopy(dict(result.qc))
    return {"value": str(result.qc)}


def _recipe_dict(result: PipelineResult) -> dict[str, Any]:
    recipe_dict = getattr(result, "recipe_dict", None)
    if callable(recipe_dict):
        return recipe_dict()
    return deepcopy(dict(result.recipe))


def _build_prepared(
    result: PipelineResult,
    *,
    spectra: ArrayLike,
    normalization_state: NormalizationState,
    baseline_run_id: str | None,
    fingerprint: str,
    recipe_updates: Mapping[str, Any],
    additional_warnings: tuple[str, ...] = (),
) -> PreparedSpectralDataset:
    selected = result.absorbance_selected
    recipe = _recipe_dict(result)
    recipe.update(deepcopy(dict(recipe_updates)))
    current_direction = wavenumber_direction(selected.wavenumber)
    prepared_hash = prepared_data_sha256(
        selected.wavenumber,
        selected.perturbation,
        selected.perturbation_labels,
        spectra,
        normalization_state=normalization_state,
    )
    return PreparedSpectralDataset(
        wavenumber=selected.wavenumber,
        perturbation=selected.perturbation,
        perturbation_labels=selected.perturbation_labels,
        spectra=np.asarray(spectra, dtype=np.float64),
        intensity_unit="absorbance",
        source_name=result.raw_input.source_name,
        source_sha256=result.input_sha256,
        baseline_run_id=_baseline_run_id(baseline_run_id),
        baseline_fingerprint=fingerprint,
        prepared_data_sha256=prepared_hash,
        original_axis_direction=_original_axis_direction(result),
        current_axis_direction=current_direction,
        perturbation_order_policy=_perturbation_policy(result),
        baseline_recipe=recipe,
        baseline_qc=_qc_dict(result),
        warnings=tuple(dict.fromkeys((*result.warnings, *additional_warnings))),
        normalization_state=normalization_state,
    )


def prepared_from_baseline_result(
    result: PipelineResult,
    *,
    baseline_run_id: str | None = None,
) -> PreparedSpectralDataset:
    """Create the default 2D handoff from ``PipelineResult.analysis_data``.

    This is the only general baseline adapter.  It deliberately has no data
    selector argument, which prevents display-normalized data from entering the
    primary scientific path.
    """

    if not isinstance(result, PipelineResult):
        raise TypeError("result must be ftir_baseline.pipeline.PipelineResult")
    spectra = result.analysis_data
    normalization_state = _normalization_state_for_primary(result)
    return _build_prepared(
        result,
        spectra=spectra,
        normalization_state=normalization_state,
        baseline_run_id=baseline_run_id,
        fingerprint=baseline_fingerprint(result),
        recipe_updates={
            "prepared_data_contract": {
                "source_channel": "PipelineResult.analysis_data",
                "scientific_normalization": False,
            }
        },
    )


def prepared_scientific_branch_from_baseline_result(
    result: PipelineResult,
    *,
    baseline_run_id: str | None = None,
    branch_name: str | None = None,
    normalized_spectra: ArrayLike | None = None,
    normalization_method: str | None = None,
) -> PreparedSpectralDataset:
    """Create an explicitly named scientific-normalization sensitivity branch.

    No normalization formula is implemented here.  By default this consumes the
    baseline core's ``optional_normalized`` output.  A caller may instead supply
    externally computed ``normalized_spectra``, but must explicitly name its
    method.  The display channel is never consulted.
    """

    if not isinstance(result, PipelineResult):
        raise TypeError("result must be ftir_baseline.pipeline.PipelineResult")
    resolved_method: str
    if normalized_spectra is None:
        normalized_spectra = result.normalization.optional_normalized
        resolved_method = result.normalization.method
        if normalized_spectra is None or resolved_method in {"none", "minmax_display"}:
            raise ValueError(
                "the baseline result has no explicit scientific normalization branch"
            )
        if normalization_method is not None and normalization_method != resolved_method:
            raise ValueError(
                "normalization_method does not match the baseline normalization result"
            )
    else:
        if not isinstance(normalization_method, str) or not normalization_method.strip():
            raise ValueError(
                "normalization_method is required with externally supplied normalized_spectra"
            )
        resolved_method = normalization_method
    resolved_method = resolved_method.strip()
    effective_name = (
        resolved_method if branch_name is None else str(branch_name).strip()
    )
    if not effective_name:
        raise ValueError("branch_name must not be empty")
    descriptor = {
        "kind": "scientific_normalization_sensitivity",
        "name": effective_name,
        "method": resolved_method,
    }
    return _build_prepared(
        result,
        spectra=normalized_spectra,
        normalization_state="scientific_explicit",
        baseline_run_id=baseline_run_id,
        fingerprint=baseline_fingerprint(
            result,
            scientific_branch=descriptor,
            spectra=normalized_spectra,
        ),
        recipe_updates={"prepared_data_contract": descriptor},
        additional_warnings=(
            "This prepared dataset is an explicit scientific-normalization sensitivity "
            f"branch ({resolved_method}); it must not replace the unnormalized primary 2D "
            "result.",
        ),
    )


# Short alias for callers that already describe the source object as a result.
to_prepared_dataset = prepared_from_baseline_result


__all__ = [
    "prepared_from_baseline_result",
    "prepared_scientific_branch_from_baseline_result",
    "to_prepared_dataset",
]
