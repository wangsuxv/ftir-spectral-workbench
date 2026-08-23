"""Baseline-only orchestration over the authoritative ``ftir_baseline`` core."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ftir_baseline.config import PipelineConfig
from ftir_baseline.models import SpectrumSet
from ftir_baseline.pipeline import PipelineResult, run_pipeline

from ..adapters import prepared_from_baseline_result
from ..export import build_baseline_bundle, export_baseline_bundle
from ..models import PreparedSpectralDataset


class BaselineWorkflowService:
    """Run, prepare, and export a complete baseline-only workflow.

    This module deliberately has no dependency on :mod:`ftir2dcos`.  Finishing
    and exporting a baseline run is a successful terminal workflow, while the
    returned :class:`PreparedSpectralDataset` is an optional downstream handoff.
    """

    def __init__(
        self,
        *,
        pipeline_runner: Callable[..., PipelineResult] = run_pipeline,
        prepared_adapter: Callable[..., PreparedSpectralDataset] = (
            prepared_from_baseline_result
        ),
    ) -> None:
        self._pipeline_runner = pipeline_runner
        self._prepared_adapter = prepared_adapter
        self._last_result: PipelineResult | None = None
        self._last_prepared: PreparedSpectralDataset | None = None

    @property
    def last_result(self) -> PipelineResult | None:
        """Most recent completed baseline result, if this instance has run one."""

        return self._last_result

    def run(
        self,
        data: SpectrumSet,
        config: PipelineConfig | Mapping[str, Any] | str,
        *,
        peak_regions: Sequence[Any] | None = None,
    ) -> PipelineResult:
        """Execute only the authoritative baseline pipeline."""

        if peak_regions is None:
            result = self._pipeline_runner(data, config)
        else:
            result = self._pipeline_runner(data, config, peak_regions=peak_regions)
        if not isinstance(result, PipelineResult):
            raise TypeError("baseline pipeline runner must return PipelineResult")
        self._last_result = result
        self._last_prepared = None
        return result

    def _resolve_result(self, result: PipelineResult | None) -> PipelineResult:
        resolved = self._last_result if result is None else result
        if resolved is None:
            raise RuntimeError("no completed baseline run is available")
        if not isinstance(resolved, PipelineResult):
            raise TypeError("result must be an ftir_baseline.pipeline.PipelineResult")
        return resolved

    def prepared(
        self,
        result: PipelineResult | None = None,
        *,
        baseline_run_id: str | None = None,
    ) -> PreparedSpectralDataset:
        """Create the immutable 2D-ready handoff from ``analysis_data``."""

        resolved = self._resolve_result(result)
        if (
            result is None
            and baseline_run_id is None
            and self._last_prepared is not None
        ):
            return self._last_prepared
        prepared = self._prepared_adapter(
            resolved,
            baseline_run_id=baseline_run_id,
        )
        if result is None or resolved is self._last_result:
            self._last_prepared = prepared
        return prepared

    def export_baseline_only(
        self,
        result: PipelineResult | None = None,
        *,
        prepared: PreparedSpectralDataset | None = None,
        qc_figures: Mapping[str, bytes] | None = None,
        destination: str | Path | None = None,
    ) -> bytes | Path:
        """Finish the workflow with a standalone, verifiable baseline bundle."""

        resolved = self._resolve_result(result)
        prepared_value = prepared
        if prepared_value is None:
            prepared_value = self.prepared(
                None if resolved is self._last_result else resolved
            )
        return export_baseline_bundle(
            resolved,
            destination,
            prepared=prepared_value,
            qc_figures=qc_figures,
        )

    def build_baseline_only_bundle(
        self,
        result: PipelineResult | None = None,
        *,
        prepared: PreparedSpectralDataset | None = None,
        qc_figures: Mapping[str, bytes] | None = None,
    ) -> bytes:
        """Explicit in-memory spelling of :meth:`export_baseline_only`."""

        resolved = self._resolve_result(result)
        prepared_value = prepared or self.prepared(
            None if resolved is self._last_result else resolved
        )
        return build_baseline_bundle(
            resolved,
            prepared=prepared_value,
            qc_figures=qc_figures,
        )

    # Short UI/CLI-facing alias.  It still means baseline-only export.
    export = export_baseline_only


__all__ = ["BaselineWorkflowService"]
