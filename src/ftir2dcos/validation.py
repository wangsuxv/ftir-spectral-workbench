"""Dataset quality checks and explicitly recorded ordering/range operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .models import SpectralDataset, ValidationReport

ORDER_MODES = ("preserve_file_order", "sort_by_perturbation")


def wavenumber_direction(wavenumber: np.ndarray) -> str:
    """Classify an axis as ascending, descending, or invalid/non-monotonic."""

    values = np.asarray(wavenumber, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("wavenumber must be one-dimensional")
    if values.size < 2 or not np.all(np.isfinite(values)):
        return "undetermined"
    differences = np.diff(values)
    if np.all(differences > 0):
        return "ascending"
    if np.all(differences < 0):
        return "descending"
    return "non_monotonic"


def estimate_2d_matrix_memory(n_wavenumbers: int) -> dict[str, int]:
    """Return exact float64 byte estimates for one and two square matrices."""

    n = int(n_wavenumbers)
    if n < 0:
        raise ValueError("n_wavenumbers cannot be negative")
    single = n * n * np.dtype(np.float64).itemsize
    return {"single_matrix_bytes": single, "two_matrix_bytes": 2 * single}


def _finite_min_max(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return float(np.min(finite)), float(np.max(finite))


def _duplicates(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return []
    unique, counts = np.unique(finite, return_counts=True)
    return [float(value) for value in unique[counts > 1]]


def validate_dataset(
    dataset: SpectralDataset,
    wavenumber_range: tuple[float, float] | None = None,
    *,
    low_wavenumber: float | None = None,
    high_wavenumber: float | None = None,
    minimum_wavenumber_points: int = 10,
    equal_spacing_rtol: float = 1e-5,
) -> ValidationReport:
    """Validate all parts of the input contract without modifying the dataset.

    Blocking data defects are collected in ``errors``.  Scientifically relevant
    but computable conditions (few spectra, duplicate spectra, non-monotonic or
    non-uniform perturbations) are reported as warnings and never silently fixed.
    """

    errors: list[str] = []
    warnings: list[str] = [str(item) for item in dataset.metadata.get("parse_warnings", [])]
    metrics: dict[str, Any] = {
        "n_spectra": dataset.n_spectra,
        "n_wavenumbers": dataset.n_wavenumbers,
        "spectra_dtype": str(dataset.spectra.dtype),
        "wavenumber_dtype": str(dataset.wavenumber.dtype),
        "perturbation_dtype": str(dataset.perturbation.dtype),
    }

    x = np.asarray(dataset.wavenumber, dtype=np.float64)
    spectra = np.asarray(dataset.spectra, dtype=np.float64)
    perturbation = np.asarray(dataset.perturbation, dtype=np.float64)

    minimum_points = int(minimum_wavenumber_points)
    if minimum_points < 1:
        raise ValueError("minimum_wavenumber_points must be positive")
    if x.size < minimum_points:
        errors.append(
            f"Wavenumber axis has {x.size} points; at least {minimum_points} are required."
        )
    nonfinite_x = np.flatnonzero(~np.isfinite(x))
    metrics["wavenumber_nonfinite_count"] = int(nonfinite_x.size)
    if nonfinite_x.size:
        errors.append(
            "Wavenumber axis contains NaN or Inf at indices "
            f"{nonfinite_x[:10].tolist()}. No values were removed."
        )
    duplicate_wavenumbers = _duplicates(x)
    metrics["duplicate_wavenumbers"] = duplicate_wavenumbers
    if duplicate_wavenumbers:
        errors.append(
            "Wavenumber axis contains duplicate values "
            f"{duplicate_wavenumbers[:10]}. No duplicates were removed."
        )

    direction = wavenumber_direction(x)
    metrics["wavenumber_direction"] = direction
    if direction == "non_monotonic":
        errors.append("Wavenumber axis is not strictly monotonic. No sorting was performed.")
    elif direction == "undetermined" and x.size >= 2 and not nonfinite_x.size:
        errors.append("Wavenumber direction could not be determined.")
    elif direction == "descending":
        warnings.append(
            "Wavenumber axis is descending; conversion to ascending must be explicit and recorded."
        )

    x_min, x_max = _finite_min_max(x)
    metrics["wavenumber_min"] = x_min
    metrics["wavenumber_max"] = x_max
    if x.size >= 2 and np.all(np.isfinite(x)):
        intervals = np.diff(x)
        absolute_intervals = np.abs(intervals)
        median_step = float(np.median(intervals))
        median_absolute_step = float(np.median(absolute_intervals))
        mean_absolute_step = float(np.mean(absolute_intervals))
        interval_cv = (
            float(np.std(absolute_intervals) / mean_absolute_step)
            if mean_absolute_step > 0
            else None
        )
        metrics.update(
            {
                "wavenumber_median_step": median_step,
                "wavenumber_median_absolute_step": median_absolute_step,
                "wavenumber_interval_cv": interval_cv,
            }
        )

    nan_locations = np.argwhere(np.isnan(spectra))
    inf_locations = np.argwhere(np.isinf(spectra))
    metrics["spectra_nan_count"] = int(nan_locations.shape[0])
    metrics["spectra_inf_count"] = int(inf_locations.shape[0])
    if nan_locations.size:
        errors.append(
            "Spectra contain NaN at (spectrum, wavenumber-index) positions "
            f"{nan_locations[:10].tolist()}. No rows or values were removed."
        )
    if inf_locations.size:
        errors.append(
            "Spectra contain Inf at (spectrum, wavenumber-index) positions "
            f"{inf_locations[:10].tolist()}. No rows or values were removed."
        )

    duplicate_spectrum_groups: list[list[int]] = []
    constant_spectra: list[int] = []
    if np.all(np.isfinite(spectra)):
        _, inverse, counts = np.unique(spectra, axis=0, return_inverse=True, return_counts=True)
        for group_index in np.flatnonzero(counts > 1):
            duplicate_spectrum_groups.append(np.flatnonzero(inverse == group_index).tolist())
        constant_spectra = [
            index for index, row in enumerate(spectra) if row.size and bool(np.all(row == row[0]))
        ]
    metrics["duplicate_spectrum_groups"] = duplicate_spectrum_groups
    metrics["constant_spectra_indices"] = constant_spectra
    if duplicate_spectrum_groups:
        warnings.append(
            "Completely identical spectra were found at index groups "
            f"{duplicate_spectrum_groups}; they were preserved."
        )
    if constant_spectra:
        warnings.append(
            f"Constant spectra were found at indices {constant_spectra}; they were preserved."
        )
    if dataset.n_spectra < 3:
        errors.append(
            f"Only {dataset.n_spectra} spectra are present; at least 3 are required for 2D-COS."
        )
    elif dataset.n_spectra < 5:
        warnings.append(
            f"Only {dataset.n_spectra} spectra are present; 2D-COS results may be unstable."
        )

    nonfinite_perturbation = np.flatnonzero(~np.isfinite(perturbation))
    metrics["perturbation_nonfinite_indices"] = nonfinite_perturbation.tolist()
    if nonfinite_perturbation.size:
        labels = [dataset.perturbation_labels[index] for index in nonfinite_perturbation]
        errors.append(
            "Perturbation values are missing or non-finite for labels "
            f"{labels}. Supply explicit numeric values; file order was preserved."
        )
    duplicate_perturbations = _duplicates(perturbation)
    metrics["duplicate_perturbations"] = duplicate_perturbations
    if duplicate_perturbations:
        warnings.append(
            f"Perturbation values contain duplicates {duplicate_perturbations}; they were preserved."
        )

    if perturbation.size >= 2 and np.all(np.isfinite(perturbation)):
        differences = np.diff(perturbation)
        if np.all(differences > 0):
            perturbation_direction = "ascending"
        elif np.all(differences < 0):
            perturbation_direction = "descending"
        else:
            perturbation_direction = "non_monotonic"
            warnings.append(
                "Perturbation values are not strictly monotonic in the current file order; "
                "no sorting was performed."
            )
        absolute_differences = np.abs(differences)
        reference_interval = float(np.median(absolute_differences))
        equal_spacing_atol = max(reference_interval * 1e-12, np.finfo(np.float64).eps)
        approximately_equal = bool(
            np.allclose(
                absolute_differences,
                reference_interval,
                rtol=float(equal_spacing_rtol),
                atol=equal_spacing_atol,
            )
        )
        metrics.update(
            {
                "perturbation_direction": perturbation_direction,
                "perturbation_intervals": differences.astype(float).tolist(),
                "perturbation_approximately_equally_spaced": approximately_equal,
                "hilbert_grid_strategy": "index_order",
            }
        )
        if not approximately_equal:
            warnings.append(
                "Perturbation intervals are non-uniform: "
                f"{differences.astype(float).tolist()}. The first release constructs the "
                "Hilbert-Noda matrix from acquisition indices, without time-weighting, "
                "interpolation, or resampling."
            )

    if wavenumber_range is not None and (low_wavenumber is not None or high_wavenumber is not None):
        raise ValueError("Specify wavenumber_range or low/high_wavenumber keyword values, not both")
    requested_range = wavenumber_range
    if requested_range is None and (low_wavenumber is not None or high_wavenumber is not None):
        if low_wavenumber is None or high_wavenumber is None:
            raise ValueError("Both low_wavenumber and high_wavenumber are required")
        requested_range = (low_wavenumber, high_wavenumber)
    if requested_range is not None:
        if len(requested_range) != 2:
            raise ValueError("wavenumber_range must contain exactly two values")
        lower = float(min(requested_range))
        upper = float(max(requested_range))
        metrics["requested_wavenumber_range"] = [lower, upper]
        if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
            errors.append("Requested wavenumber range must contain two distinct finite values.")
        elif x_min is not None and x_max is not None:
            if lower < x_min or upper > x_max:
                errors.append(
                    f"Requested range [{lower}, {upper}] is outside the available range "
                    f"[{x_min}, {x_max}]."
                )
            selected_count = int(np.count_nonzero((x >= lower) & (x <= upper)))
            metrics["selected_wavenumber_points"] = selected_count
            if selected_count < minimum_points:
                errors.append(
                    f"Requested range contains {selected_count} points; at least "
                    f"{minimum_points} are required."
                )

    memory = estimate_2d_matrix_memory(dataset.n_wavenumbers)
    metrics.update(memory)
    if dataset.n_wavenumbers > 2500:
        warnings.append(
            f"The dataset has {dataset.n_wavenumbers} wavenumber points; each float64 2D "
            f"matrix needs {memory['single_matrix_bytes']} bytes and matrix CSV files may be large."
        )

    # Preserve issue order while removing repeated parse/validation messages.
    return ValidationReport(
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        metrics=metrics,
    )


def _metadata_with_history(
    dataset: SpectralDataset,
    operation: dict[str, Any],
    **updates: Any,
) -> dict[str, Any]:
    metadata = deepcopy(dict(dataset.metadata))
    history = list(metadata.get("processing_history", []))
    history.append(deepcopy(operation))
    metadata["processing_history"] = history
    metadata.update(deepcopy(updates))
    return metadata


def apply_perturbation_order(
    dataset: SpectralDataset,
    mode: str = "preserve_file_order",
) -> SpectralDataset:
    """Preserve or stably sort spectra, always recording the exact index mapping."""

    normalized_mode = str(mode).strip().lower().replace("-", "_")
    if normalized_mode not in ORDER_MODES:
        raise ValueError(f"mode must be one of {ORDER_MODES}; got {mode!r}")
    original_indices = np.arange(dataset.n_spectra, dtype=np.int64)
    if normalized_mode == "preserve_file_order":
        indices = original_indices
    else:
        if not np.all(np.isfinite(dataset.perturbation)):
            raise ValueError(
                "Cannot sort by perturbation while values contain NaN or Inf; supply numeric values first"
            )
        indices = np.argsort(dataset.perturbation, kind="stable")

    changed = not np.array_equal(indices, original_indices)
    original_labels = list(dataset.perturbation_labels)
    final_labels = [dataset.perturbation_labels[index] for index in indices]
    metadata = _metadata_with_history(
        dataset,
        {
            "operation": "apply_perturbation_order",
            "mode": normalized_mode,
            "original_indices": original_indices.tolist(),
            "final_original_indices": indices.tolist(),
            "order_changed": changed,
        },
        perturbation_order_mode=normalized_mode,
        perturbation_original_labels=dataset.metadata.get(
            "perturbation_original_labels", original_labels
        ),
        perturbation_final_labels=final_labels,
        perturbation_original_values=dataset.perturbation.astype(float).tolist(),
        perturbation_final_values=dataset.perturbation[indices].astype(float).tolist(),
        perturbation_order_indices=indices.tolist(),
        perturbation_order_changed=changed,
    )
    return dataset.with_updates(
        perturbation=dataset.perturbation[indices],
        perturbation_labels=tuple(final_labels),
        spectra=dataset.spectra[indices, :],
        metadata=metadata,
    )


def ensure_ascending_wavenumber(dataset: SpectralDataset) -> SpectralDataset:
    """Return an ascending-axis dataset, rejecting rather than sorting bad axes."""

    direction = wavenumber_direction(dataset.wavenumber)
    if direction not in {"ascending", "descending"}:
        raise ValueError(
            f"Cannot orient a {direction} wavenumber axis. Fix non-finite, duplicate, or "
            "non-monotonic input explicitly."
        )
    if direction == "descending":
        indices = np.arange(dataset.n_wavenumbers - 1, -1, -1)
    else:
        indices = np.arange(dataset.n_wavenumbers)
    changed = direction == "descending"
    metadata = _metadata_with_history(
        dataset,
        {
            "operation": "ensure_ascending_wavenumber",
            "original_direction": direction,
            "final_direction": "ascending",
            "order_changed": changed,
        },
        original_wavenumber_direction=dataset.metadata.get(
            "original_wavenumber_direction", direction
        ),
        wavenumber_direction="ascending",
        wavenumber_order_changed=bool(
            dataset.metadata.get("wavenumber_order_changed", False) or changed
        ),
    )
    return dataset.with_updates(
        wavenumber=dataset.wavenumber[indices],
        spectra=dataset.spectra[:, indices],
        metadata=metadata,
    )


def select_wavenumber_range(
    dataset: SpectralDataset,
    value1: float | None = None,
    value2: float | None = None,
    *,
    low_wavenumber: float | None = None,
    high_wavenumber: float | None = None,
    ensure_ascending: bool = True,
    minimum_points: int = 10,
) -> SpectralDataset:
    """Select an inclusive range after an explicit, recorded direction choice."""

    if value1 is not None or value2 is not None:
        if low_wavenumber is not None or high_wavenumber is not None:
            raise ValueError("Use positional range values or low/high keywords, not both")
        if value1 is None or value2 is None:
            raise ValueError("Two wavenumber range values are required")
        first, second = float(value1), float(value2)
    else:
        if low_wavenumber is None or high_wavenumber is None:
            raise ValueError("Both low_wavenumber and high_wavenumber are required")
        first, second = float(low_wavenumber), float(high_wavenumber)
    if not np.isfinite(first) or not np.isfinite(second) or first == second:
        raise ValueError("Wavenumber bounds must be distinct finite values")
    lower, upper = min(first, second), max(first, second)

    oriented = ensure_ascending_wavenumber(dataset) if ensure_ascending else dataset.with_updates()
    report = validate_dataset(
        oriented,
        (lower, upper),
        minimum_wavenumber_points=minimum_points,
    )
    report.raise_for_errors()
    mask = (oriented.wavenumber >= lower) & (oriented.wavenumber <= upper)
    selected_x = oriented.wavenumber[mask]
    memory = estimate_2d_matrix_memory(selected_x.size)
    processing_warnings = list(oriented.metadata.get("processing_warnings", []))
    if selected_x.size > 2500:
        processing_warnings.append(
            f"Selected range has {selected_x.size} points; two float64 2D matrices require "
            f"approximately {memory['two_matrix_bytes']} bytes before export overhead."
        )
    metadata = _metadata_with_history(
        oriented,
        {
            "operation": "select_wavenumber_range",
            "requested_bounds": [first, second],
            "normalized_bounds": [lower, upper],
            "inclusive": True,
            "selected_points": int(selected_x.size),
            "interpolation_performed": False,
        },
        selected_wavenumber_range=[lower, upper],
        selected_actual_wavenumber_range=[float(selected_x[0]), float(selected_x[-1])],
        selected_wavenumber_points=int(selected_x.size),
        selected_matrix_memory=memory,
        processing_warnings=processing_warnings,
    )
    return oriented.with_updates(
        wavenumber=selected_x,
        spectra=oriented.spectra[:, mask],
        metadata=metadata,
    )


# Naming aliases used by the UI/pipeline layers.
orient_wavenumber_ascending = ensure_ascending_wavenumber
crop_wavenumber_range = select_wavenumber_range


__all__ = [
    "ORDER_MODES",
    "apply_perturbation_order",
    "crop_wavenumber_range",
    "ensure_ascending_wavenumber",
    "estimate_2d_matrix_memory",
    "orient_wavenumber_ascending",
    "select_wavenumber_range",
    "validate_dataset",
    "wavenumber_direction",
]
