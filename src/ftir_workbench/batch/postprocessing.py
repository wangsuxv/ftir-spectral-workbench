"""Ordinary B -> S -> N_S and B -> N_B; immutable, independently committed branches.

Only the two array APIs perform numerical work. Parent resolution, state queries,
confirmation and export validation never fit a baseline or create Prepared data.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ftir_baseline.models import FloatArray, freeze_value, immutable_float64
from ftir_workbench.display_units import DisplayIntensityUnit
from ftir_workbench.post_baseline_smoothing import (
    PostBaselineSmoothingConfig,
    effective_smoothing_config,
    smooth_spectral_arrays,
    validate_smoothing_request,
    validate_spectral_arrays,
)

from .fingerprints import json_fingerprint
from .models import BatchError, BatchWorkspace, StageSnapshot
from .normalization_adapter import (
    OrdinaryNormalizationConfig,
    normalize_config,
    normalize_spectral_arrays,
    validate_normalization_request,
)
from .state import _validate_snapshot, get_ready_snapshot

BRANCHES = ("baseline", "smoothed", "normalized_baseline", "normalized_smoothed")
DERIVED_BRANCHES = BRANCHES[1:]
PARENTS = {"smoothed": "baseline", "normalized_baseline": "baseline", "normalized_smoothed": "smoothed"}


def plain(value: Any) -> Any:
    """Lossless finite JSON values, including immutable scientific metadata."""
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    return value


def draft_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    # JSON forbids executable/pickle objects and nonfinite unfinished numbers.
    # UI stores an unfinished numeric control as null, never NaN.
    try:
        return dict(json.loads(json.dumps(plain(value), allow_nan=False)))
    except (TypeError, ValueError) as exc:
        raise BatchError("POSTPROCESS_DRAFT_INVALID", "draft must contain finite JSON values") from exc


def array_hash(value: FloatArray) -> str:
    array = np.asarray(value, dtype="<f8")
    digest = hashlib.sha256(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "ftir-spectral-workbench")
    }}


@lru_cache(maxsize=1)
def postprocessing_implementation() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [root / "batch/postprocessing.py", root / "batch/normalization_adapter.py",
             root / "post_baseline_smoothing.py", root.parent / "ftir_baseline/normalization.py"]
    return json_fingerprint({
        "contract": "ordinary-postprocess-v1",
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "numerical_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
        "python": platform.python_version(),
    })


@dataclass(frozen=True, slots=True)
class PostprocessSnapshot:
    workspace_id: str
    spectrum_id: str
    source_id: str
    input_sha256: str
    branch: str
    baseline: StageSnapshot
    parent_smoothed: PostprocessSnapshot | None
    parent_fingerprint: str
    recipe: Mapping[str, Any]
    effective_recipe: Mapping[str, Any]
    implementation: str
    versions: Mapping[str, str]
    request_fingerprint: str
    fingerprint: str
    wavenumber: FloatArray
    spectra: FloatArray
    quantity: str
    purpose: str
    removed_component: FloatArray | None = None
    scale: FloatArray | None = None
    offset: FloatArray | None = None
    reference_details: Mapping[str, Any] = field(default_factory=dict)
    qc: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        axis, spectra = validate_spectral_arrays(self.wavenumber, self.spectra)
        if spectra.shape[0] != 1:
            raise BatchError("POSTPROCESS_INTEGRITY_FAILED", "ordinary snapshot must contain exactly one row")
        object.__setattr__(self, "wavenumber", axis)
        object.__setattr__(self, "spectra", spectra)
        for key in ("removed_component", "scale", "offset"):
            value = getattr(self, key)
            if value is not None:
                if np.iscomplexobj(value):
                    raise BatchError("POSTPROCESS_INTEGRITY_FAILED", f"{key} must be real")
                array = immutable_float64(value, name=key)
                shape = spectra.shape if key == "removed_component" else (1,)
                if array.shape != shape or not np.isfinite(array).all():
                    raise BatchError("POSTPROCESS_INTEGRITY_FAILED", f"{key} must be finite with shape {shape}")
                object.__setattr__(self, key, array)
        for key in ("recipe", "effective_recipe", "versions", "reference_details", "qc"):
            value = draft_copy(getattr(self, key))
            object.__setattr__(self, key, freeze_value(value))
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(slots=True)
class BranchState:
    draft: dict[str, Any] = field(default_factory=dict)
    preview: PostprocessSnapshot | None = None
    committed: PostprocessSnapshot | None = None
    committed_draft: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OrdinaryPostprocessingState:
    smoothed: BranchState = field(default_factory=lambda: BranchState(PostBaselineSmoothingConfig().to_dict()))
    normalized_baseline: BranchState = field(default_factory=lambda: BranchState(OrdinaryNormalizationConfig().to_dict()))
    normalized_smoothed: BranchState = field(default_factory=lambda: BranchState(OrdinaryNormalizationConfig().to_dict()))
    normalization_source: str = "baseline"
    export_choice: str = "baseline"
    display_preferences: dict[str, Any] = field(default_factory=dict)


def get_postprocessing_state(workspace: BatchWorkspace, spectrum_id: str) -> OrdinaryPostprocessingState:
    if spectrum_id not in workspace.records:
        raise BatchError("UNKNOWN_SPECTRUM", "spectrum is not in this workspace")
    if spectrum_id not in workspace.postprocessing:
        workspace.postprocessing[spectrum_id] = OrdinaryPostprocessingState()
    return workspace.postprocessing[spectrum_id]


def branch_state(workspace: BatchWorkspace, spectrum_id: str, branch: str) -> BranchState:
    if branch not in DERIVED_BRANCHES:
        raise BatchError("POSTPROCESS_CHAIN_INVALID", "only S, N_B and N_S have editable branches")
    return getattr(get_postprocessing_state(workspace, spectrum_id), branch)


def effective_config(branch: str, draft: Mapping[str, Any]) -> PostBaselineSmoothingConfig | OrdinaryNormalizationConfig:
    if branch not in DERIVED_BRANCHES:
        raise BatchError("POSTPROCESS_CHAIN_INVALID", "unsupported derived branch")
    try:
        config = effective_smoothing_config(draft) if branch == "smoothed" else normalize_config(draft)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, BatchError):
            raise
        raise BatchError("POSTPROCESS_CONFIG_INVALID", str(exc)) from exc
    if not config.enabled:
        raise BatchError("POSTPROCESS_DISABLED", "enable the selected operation before previewing or confirming")
    return config


def update_draft(workspace: BatchWorkspace, spectrum_id: str, branch: str,
                 values: Mapping[str, Any]) -> None:
    state = branch_state(workspace, spectrum_id, branch)
    state.draft = draft_copy({**state.draft, **values})
    # Keep both preview and confirmed snapshot. Currentness is derived from their
    # actual scientific dependencies, and an old confirmed result remains usable.
    state.errors.clear()


def branch_arrays(snapshot: StageSnapshot | PostprocessSnapshot) -> tuple[FloatArray, FloatArray]:
    if isinstance(snapshot, StageSnapshot):
        return snapshot.result.absorbance_selected.wavenumber, snapshot.result.analysis_data
    return snapshot.wavenumber, snapshot.spectra


def _request_payload(snapshot: PostprocessSnapshot) -> dict[str, Any]:
    parent: StageSnapshot | PostprocessSnapshot = snapshot.parent_smoothed or snapshot.baseline
    x, y = branch_arrays(parent)
    return {
        "contract": "ordinary-postprocess-v1", "workspace_id": snapshot.workspace_id,
        "spectrum_id": snapshot.spectrum_id, "source_id": snapshot.source_id,
        "input_sha256": snapshot.input_sha256, "branch": snapshot.branch,
        "baseline_fingerprint": snapshot.baseline.fingerprint,
        "parent_fingerprint": snapshot.parent_fingerprint, "parent_branch": PARENTS[snapshot.branch],
        "x_sha256": array_hash(x), "parent_y_sha256": array_hash(y),
        "effective_recipe": plain(snapshot.effective_recipe), "implementation": snapshot.implementation,
    }


def snapshot_fingerprint(snapshot: PostprocessSnapshot) -> str:
    return json_fingerprint({
        "request": snapshot.request_fingerprint, "x": array_hash(snapshot.wavenumber),
        "y": array_hash(snapshot.spectra), "quantity": snapshot.quantity, "purpose": snapshot.purpose,
        "removed": None if snapshot.removed_component is None else array_hash(snapshot.removed_component),
        "scale": None if snapshot.scale is None else array_hash(snapshot.scale),
        "offset": None if snapshot.offset is None else array_hash(snapshot.offset),
        "reference_details": plain(snapshot.reference_details), "qc": plain(snapshot.qc),
        "warnings": list(snapshot.warnings), "versions": plain(snapshot.versions),
    })


def validate_snapshot(workspace: BatchWorkspace, spectrum_id: str, snapshot: PostprocessSnapshot) -> None:
    """Check immutable provenance/arrays without replaying either numerical step.

    Historical parents are valid for persistence even when they are no longer
    current. get_branch additionally checks the currently confirmed parent.
    """
    def fail(message: str) -> None:
        raise BatchError("POSTPROCESS_INTEGRITY_FAILED", message)

    if not isinstance(snapshot, PostprocessSnapshot) or snapshot.branch not in DERIVED_BRANCHES:
        fail("invalid derived snapshot type or branch")
    record = workspace.records[spectrum_id]
    if (snapshot.workspace_id != workspace.workspace_id or snapshot.spectrum_id != spectrum_id
            or snapshot.source_id != record.source_id or snapshot.input_sha256 != snapshot.baseline.input_sha256):
        fail("snapshot identity or source mismatch")
    _validate_snapshot(workspace, spectrum_id, snapshot.baseline)
    if snapshot.baseline.stage != "fine":
        fail("baseline parent is not an explicitly finalized baseline snapshot")
    if snapshot.branch == "normalized_smoothed":
        parent = snapshot.parent_smoothed
        if parent is None or parent.branch != "smoothed":
            fail("N_S requires its confirmed S parent")
        assert parent is not None
        validate_snapshot(workspace, spectrum_id, parent)
        if parent.baseline.fingerprint != snapshot.baseline.fingerprint:
            fail("S and N_S disagree on their B parent")
    elif snapshot.parent_smoothed is not None:
        fail("this branch must have B as its direct parent")
    parent_snapshot: StageSnapshot | PostprocessSnapshot = snapshot.parent_smoothed or snapshot.baseline
    x, y = branch_arrays(parent_snapshot)
    if (snapshot.parent_fingerprint != parent_snapshot.fingerprint
            or not np.array_equal(snapshot.wavenumber, x) or snapshot.spectra.shape != y.shape):
        fail("parent hash, axis or shape mismatch")
    config = effective_config(snapshot.branch, snapshot.recipe)
    if plain(config.scientific_dict()) != plain(snapshot.effective_recipe):
        fail("effective recipe differs from recorded active controls")
    if not snapshot.implementation or not snapshot.versions:
        fail("implementation and dependency versions are required")
    if snapshot.branch == "smoothed":
        if (snapshot.quantity != "absorbance" or snapshot.purpose != "scientific"
                or snapshot.scale is not None or snapshot.offset is not None
                or snapshot.removed_component is None
                or not np.array_equal(snapshot.removed_component, y - snapshot.spectra)):
            fail("invalid S quantity or removed component")
    else:
        expected_minmax = config.method == "minmax_display"
        if (snapshot.quantity != ("minmax_scaled_intensity" if expected_minmax else "normalized_intensity")
                or snapshot.purpose != ("display_only" if expected_minmax else "scientific")
                or snapshot.removed_component is not None or snapshot.scale is None or snapshot.offset is None):
            fail("invalid normalization quantity or affine metadata")
        assert snapshot.scale is not None and snapshot.offset is not None
        if (snapshot.scale.shape != (1,) or snapshot.offset.shape != (1,)
                or np.any(snapshot.scale <= 0) or (not expected_minmax and np.any(snapshot.offset != 0))):
            fail("normalization requires a finite positive per-spectrum scale")
        with np.errstate(over="ignore", invalid="ignore"):
            if expected_minmax:
                minimum = np.asarray(snapshot.reference_details.get("minimum"), dtype=float)
                if minimum.shape != (1,) or not np.array_equal(minimum, np.min(y, axis=1)):
                    fail("Min-Max minimum does not match its parent")
                # The core evaluates the stable translated expression, not
                # scale*y+offset which can catastrophically cancel large offsets.
                affine = (y - minimum[:, None]) * snapshot.scale[:, None]
                if not np.array_equal(snapshot.offset, -minimum / (np.max(y, axis=1) - minimum)):
                    fail("Min-Max offset does not match its scale and parent minimum")
            else:
                affine = snapshot.scale[:, None] * y
        if not np.array_equal(affine, snapshot.spectra):
            fail("normalization arrays disagree with stored scale/offset")
    if snapshot.request_fingerprint != json_fingerprint(_request_payload(snapshot)):
        fail("scientific request fingerprint mismatch")
    if snapshot.fingerprint != snapshot_fingerprint(snapshot):
        fail("output fingerprint mismatch")


def get_branch(workspace: BatchWorkspace, spectrum_id: str, branch: str = "baseline") -> StageSnapshot | PostprocessSnapshot:
    if branch not in BRANCHES:
        raise BatchError("POSTPROCESS_CHAIN_INVALID", "unsupported branch")
    baseline = get_ready_snapshot(workspace, spectrum_id, "fine")
    if branch == "baseline":
        return baseline
    snapshot = branch_state(workspace, spectrum_id, branch).committed
    if snapshot is None:
        raise BatchError("POSTPROCESS_REQUIRED", f"confirm {branch} before using it")
    validate_snapshot(workspace, spectrum_id, snapshot)
    if snapshot.baseline.fingerprint != baseline.fingerprint:
        raise BatchError("POSTPROCESS_STALE", "the confirmed final baseline parent has changed")
    if branch == "normalized_smoothed":
        current_s = get_branch(workspace, spectrum_id, "smoothed")
        if current_s.fingerprint != snapshot.parent_fingerprint:
            raise BatchError("POSTPROCESS_STALE", "the confirmed smoothing parent has changed")
    return snapshot


def _resolve_request(workspace: BatchWorkspace, spectrum_id: str, branch: str) -> PostprocessSnapshot:
    state = branch_state(workspace, spectrum_id, branch)
    config = effective_config(branch, state.draft)
    baseline = get_ready_snapshot(workspace, spectrum_id, "fine")
    parent = get_branch(workspace, spectrum_id, PARENTS[branch])
    x, y = branch_arrays(parent)
    record = workspace.records[spectrum_id]
    provisional = PostprocessSnapshot(
        workspace_id=workspace.workspace_id, spectrum_id=spectrum_id, source_id=record.source_id,
        input_sha256=baseline.input_sha256, branch=branch, baseline=baseline,
        parent_smoothed=parent if isinstance(parent, PostprocessSnapshot) else None,
        parent_fingerprint=parent.fingerprint, recipe=state.draft,
        effective_recipe=config.scientific_dict(), implementation=postprocessing_implementation(),
        versions=dependency_versions(), request_fingerprint="", fingerprint="",
        wavenumber=x, spectra=y, quantity="absorbance", purpose="scientific",
    )
    return replace(provisional, request_fingerprint=json_fingerprint(_request_payload(provisional)))


def preview_branch(workspace: BatchWorkspace, spectrum_id: str, branch: str) -> PostprocessSnapshot:
    state = branch_state(workspace, spectrum_id, branch)
    try:
        request = _resolve_request(workspace, spectrum_id, branch)
        for cached in (state.preview, state.committed):
            if cached is not None and cached.request_fingerprint == request.request_fingerprint:
                validate_snapshot(workspace, spectrum_id, cached)
                state.preview = cached if plain(cached.recipe) == plain(state.draft) else replace(cached, recipe=state.draft)
                state.errors.clear()
                return state.preview
        config = effective_config(branch, state.draft)
        if branch == "smoothed":
            raw_metadata = request.baseline.result.raw_input.metadata
            # Current ordinary imports have no multi-segment format. Explicit
            # unsupported segment annotations must not bypass the axis override.
            if raw_metadata.get("segment_boundaries") or raw_metadata.get("discontinuous_segments"):
                raise BatchError("SEGMENTED_AXIS_UNSUPPORTED", "cross-segment smoothing is unsupported")
            result = smooth_spectral_arrays(request.wavenumber, request.spectra, config.to_dict())
            snapshot = replace(
                request, spectra=result.smoothed_spectra, removed_component=result.removed_component,
                qc={"per_spectrum": plain(result.per_spectrum_metrics), "summary": plain(result.summary_metrics),
                    "median_wavenumber_spacing": result.median_wavenumber_spacing,
                    "spacing_relative_max_deviation": result.spacing_relative_max_deviation,
                    "approximate_physical_width": plain(result.approximate_physical_width)},
                warnings=result.warnings,
            )
        else:
            normalized = normalize_spectral_arrays(request.wavenumber, request.spectra, config.to_dict())
            snapshot = replace(
                request, spectra=normalized.normalized_spectra, scale=normalized.scale, offset=normalized.offset,
                reference_details=normalized.reference_details, quantity=normalized.quantity,
                purpose=normalized.purpose, qc=normalized.metrics, warnings=normalized.warnings,
            )
        snapshot = replace(snapshot, fingerprint=snapshot_fingerprint(snapshot))
        validate_snapshot(workspace, spectrum_id, snapshot)
        state.preview = snapshot
        state.errors.clear()
        return snapshot
    except (ValueError, TypeError, KeyError) as exc:
        state.errors[:] = [str(exc)]
        raise


def confirm_branch(workspace: BatchWorkspace, spectrum_id: str, branch: str) -> PostprocessSnapshot:
    state = branch_state(workspace, spectrum_id, branch)
    preview = state.preview
    if preview is None:
        raise BatchError("PREVIEW_OUTDATED", "preview this branch before confirming")
    request = _resolve_request(workspace, spectrum_id, branch)
    if preview.branch != branch or preview.request_fingerprint != request.request_fingerprint:
        raise BatchError("PREVIEW_OUTDATED", "preview no longer matches this id/source/config/parent")
    validate_snapshot(workspace, spectrum_id, preview)
    state.committed = preview if plain(preview.recipe) == plain(state.draft) else replace(preview, recipe=state.draft)
    state.committed_draft = draft_copy(state.draft)
    state.errors.clear()
    return state.committed


def branch_status(workspace: BatchWorkspace, spectrum_id: str, branch: str) -> dict[str, Any]:
    state = None if branch == "baseline" else branch_state(workspace, spectrum_id, branch)
    result: dict[str, Any] = {"status": "missing", "reason": "", "draft_modified": False, "preview_current": False}
    try:
        get_branch(workspace, spectrum_id, branch)
        result["status"] = "ready"
    except (ValueError, TypeError, KeyError) as exc:
        result["status"] = "stale" if state is not None and state.committed is not None else "missing"
        result["reason"] = str(exc)
    if state is not None:
        result["draft_modified"] = state.committed_draft is None or state.draft != state.committed_draft
        try:
            request = _resolve_request(workspace, spectrum_id, branch)
            result["preview_current"] = state.preview is not None and state.preview.request_fingerprint == request.request_fingerprint
        except (ValueError, TypeError, KeyError):
            pass
    return result


def copy_drafts(workspace: BatchWorkspace, source_spectrum_id: str, spectrum_ids: Sequence[str], *, branch: str) -> dict[str, str | None]:
    source = draft_copy(branch_state(workspace, source_spectrum_id, branch).draft)
    report: dict[str, str | None] = {}
    for sid in dict.fromkeys(spectrum_ids):
        try:
            target = branch_state(workspace, sid, branch)
            target.draft = copy.deepcopy(source)
            target.errors.clear()
            # Resolve and validate the target without applying either numerical
            # operation. Invalid copied drafts remain editable, with a reason.
            request = _resolve_request(workspace, sid, branch)
            validator = validate_smoothing_request if branch == "smoothed" else validate_normalization_request
            validator(request.wavenumber, request.spectra, target.draft)
            report[sid] = None
        except (ValueError, TypeError, KeyError) as exc:
            report[sid] = str(exc)
    return report


def _selected(workspace: BatchWorkspace, spectrum_ids: Sequence[str], branch: str, *, confirm: bool) -> dict[str, str | None]:
    report: dict[str, str | None] = {}
    for sid in dict.fromkeys(spectrum_ids):
        try:
            if workspace.records[sid].excluded:
                raise BatchError("EXCLUDED", "spectrum explicitly excluded")
            (confirm_branch if confirm else preview_branch)(workspace, sid, branch)
            report[sid] = None
        except (ValueError, TypeError, KeyError) as exc:
            report[sid] = str(exc)
    return report


def preview_selected(workspace: BatchWorkspace, spectrum_ids: Sequence[str], *, branch: str) -> dict[str, str | None]:
    return _selected(workspace, spectrum_ids, branch, confirm=False)


def confirm_selected(workspace: BatchWorkspace, spectrum_ids: Sequence[str], *, branch: str) -> dict[str, str | None]:
    return _selected(workspace, spectrum_ids, branch, confirm=True)


def preview_smoothing(workspace: BatchWorkspace, spectrum_id: str, *, source_kind: str = "baseline") -> PostprocessSnapshot:
    if source_kind != "baseline":
        raise BatchError("POSTPROCESS_CHAIN_INVALID", "smoothing accepts only finalized baseline B")
    return preview_branch(workspace, spectrum_id, "smoothed")


def preview_normalization(workspace: BatchWorkspace, spectrum_id: str, *, source_kind: str = "baseline") -> PostprocessSnapshot:
    if source_kind not in {"baseline", "smoothed"}:
        raise BatchError("POSTPROCESS_CHAIN_INVALID", "normalization accepts only B or S; repeated normalization is forbidden")
    return preview_branch(workspace, spectrum_id, "normalized_" + source_kind)


def display_arrays(snapshot: StageSnapshot | PostprocessSnapshot, unit: DisplayIntensityUnit = "absorbance") -> tuple[FloatArray, FloatArray, str]:
    """Quantity gate before reusing the unchanged absorbance display conversion."""
    x, y = branch_arrays(snapshot)
    if isinstance(snapshot, PostprocessSnapshot) and snapshot.branch.startswith("normalized_"):
        if unit != "absorbance":
            raise BatchError("NORMALIZED_UNIT_INVALID", "normalized/scaled intensity cannot be converted to physical T or %T")
        label = "Min-Max scaled intensity (0-1, display only)" if snapshot.purpose == "display_only" else "Normalized intensity"
        return x, y, label
    from ftir_workbench.display_units import convert_absorbance_for_display
    result = convert_absorbance_for_display(y, unit)
    return x, result.values, "Absorbance" if unit == "absorbance" else f"Derived {unit}"
