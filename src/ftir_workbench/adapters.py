"""In-memory adapters between the authoritative baseline and 2D packages."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike

from ftir_baseline.models import thaw_mapping
from ftir_baseline.pipeline import PipelineResult

from .fingerprints import baseline_fingerprint, prepared_data_sha256
from .models import AxisDirection, NormalizationState, PreparedSpectralDataset
from .post_baseline_smoothing import PostBaselineSmoothingResult
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


def prepared_from_smoothed_result(
    result: PostBaselineSmoothingResult,
) -> PreparedSpectralDataset:
    """Create an explicit Prepared child from one post-baseline smoothing result.

    The parent baseline lineage and QC remain unchanged.  Spectra are taken from
    the smoothing result, the Prepared-data dependency hash is recomputed, and
    the recipe gains an auditable description of the new scientific branch.
    """

    if not isinstance(result, PostBaselineSmoothingResult):
        raise TypeError("result must be a PostBaselineSmoothingResult")
    if not result.config.enabled:
        raise ValueError(
            "creating a smoothed Prepared branch requires smoothing to be enabled"
        )
    parent = result.parent_prepared
    if parent.normalization_state == "scientific_explicit":
        raise ValueError(
            "v0.2.5 does not combine scientific normalization and post-baseline "
            "smoothing. Select the primary unnormalized Prepared branch."
        )
    parent_contract = parent.baseline_recipe.get("prepared_data_contract")
    if "post_baseline_smoothing" in parent.baseline_recipe or (
        isinstance(parent_contract, Mapping)
        and parent_contract.get("branch_kind") == "post_baseline_smoothing"
    ):
        raise ValueError(
            "The selected Prepared dataset is already a post-baseline smoothing branch. "
            "Chained smoothing is disabled in v0.2.5."
        )
    scientific_config = result.config.scientific_dict()
    parameters = scientific_config.get("parameters")
    if not isinstance(parameters, Mapping):  # defensive: enabled config always has this
        raise ValueError("enabled smoothing config must define effective parameters")
    recipe = thaw_mapping(parent.baseline_recipe)
    previous_contract = deepcopy(recipe.get("prepared_data_contract"))
    summary_metrics = {
        str(name): float(value) for name, value in result.summary_metrics.items()
    }
    per_spectrum_metrics = {
        str(name): np.asarray(values, dtype=np.float64).tolist()
        for name, values in result.per_spectrum_metrics.items()
    }
    effective_parameters = deepcopy(dict(parameters))
    recipe["prepared_data_contract"] = {
        "source_channel": "parent PreparedSpectralDataset.spectra",
        "branch_kind": "post_baseline_smoothing",
        "parent_prepared_data_sha256": parent.prepared_data_sha256,
        "smoothing_fingerprint": result.smoothing_fingerprint,
        "algorithm": result.config.method,
        "parameters": effective_parameters,
        "nonuniform_axis_policy": result.config.nonuniform_axis_policy,
    }
    smoothing_recipe: dict[str, Any] = {
        "schema": "ftir-workbench-post-baseline-smoothing-v1",
        "parent_prepared_data_sha256": parent.prepared_data_sha256,
        "smoothing_fingerprint": result.smoothing_fingerprint,
        "method": result.config.method,
        "parameters": deepcopy(effective_parameters),
        "config": scientific_config,
        "nonuniform_axis_policy": result.config.nonuniform_axis_policy,
        "axis_diagnostics": {
            "median_wavenumber_spacing": result.median_wavenumber_spacing,
            "spacing_relative_max_deviation": (
                result.spacing_relative_max_deviation
            ),
            "approximate_physical_width": dict(result.approximate_physical_width),
        },
        "summary_metrics": summary_metrics,
        "per_spectrum_metrics": per_spectrum_metrics,
        "warnings": list(result.warnings),
    }
    if previous_contract is not None:
        smoothing_recipe["parent_prepared_data_contract"] = previous_contract
    recipe["post_baseline_smoothing"] = smoothing_recipe

    prepared_hash = prepared_data_sha256(
        parent.wavenumber,
        parent.perturbation,
        parent.perturbation_labels,
        result.smoothed_spectra,
        normalization_state=parent.normalization_state,
    )
    branch_warning = (
        "This Prepared dataset is an explicit post-baseline smoothing scientific "
        f"branch ({result.config.method}); the primary unsmoothed Prepared remains "
        "the reference branch."
    )
    warnings = tuple(
        dict.fromkeys((*parent.warnings, branch_warning, *result.warnings))
    )
    return PreparedSpectralDataset(
        wavenumber=parent.wavenumber,
        perturbation=parent.perturbation,
        perturbation_labels=parent.perturbation_labels,
        spectra=result.smoothed_spectra,
        intensity_unit="absorbance",
        source_name=parent.source_name,
        source_sha256=parent.source_sha256,
        baseline_run_id=parent.baseline_run_id,
        baseline_fingerprint=parent.baseline_fingerprint,
        prepared_data_sha256=prepared_hash,
        original_axis_direction=parent.original_axis_direction,
        current_axis_direction=parent.current_axis_direction,
        perturbation_order_policy=parent.perturbation_order_policy,
        baseline_recipe=recipe,
        baseline_qc=parent.baseline_qc,
        warnings=warnings,
        normalization_state=parent.normalization_state,
    )


# Short alias for callers that already describe the source object as a result.
to_prepared_dataset = prepared_from_baseline_result


__all__ = [
    "prepared_from_baseline_result",
    "prepared_from_smoothed_result",
    "prepared_scientific_branch_from_baseline_result",
    "to_prepared_dataset",
]
