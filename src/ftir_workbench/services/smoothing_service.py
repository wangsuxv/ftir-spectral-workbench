"""Application service for an explicit post-baseline smoothing branch."""

from __future__ import annotations

from collections.abc import Callable

from ..adapters import prepared_from_smoothed_result
from ..models import PreparedSpectralDataset
from ..post_baseline_smoothing import (
    PostBaselineSmoothingConfig,
    PostBaselineSmoothingResult,
    apply_post_baseline_smoothing,
)
from ..smoothing_export import build_smoothing_bundle

SmoothingCore = Callable[
    [PreparedSpectralDataset, PostBaselineSmoothingConfig],
    PostBaselineSmoothingResult,
]
SmoothedPreparedAdapter = Callable[
    [PostBaselineSmoothingResult],
    PreparedSpectralDataset,
]
SmoothingBundleBuilder = Callable[
    [PostBaselineSmoothingResult, PreparedSpectralDataset],
    bytes,
]


class PostBaselineSmoothingService:
    """Preview or commit one smoothing operation through the same core API.

    The service remains independent of 2D-COS.  ``preview`` returns only the
    immutable transformation result; ``apply`` additionally creates, but never
    activates, its Prepared child.  Export occurs only through the explicit
    ``build_bundle`` method.
    """

    def __init__(
        self,
        *,
        smoothing_core: SmoothingCore = apply_post_baseline_smoothing,
        prepared_adapter: SmoothedPreparedAdapter = prepared_from_smoothed_result,
        bundle_builder: SmoothingBundleBuilder = build_smoothing_bundle,
    ) -> None:
        if not callable(smoothing_core):
            raise TypeError("smoothing_core must be callable")
        if not callable(prepared_adapter):
            raise TypeError("prepared_adapter must be callable")
        if not callable(bundle_builder):
            raise TypeError("bundle_builder must be callable")
        self._smoothing_core = smoothing_core
        self._prepared_adapter = prepared_adapter
        self._bundle_builder = bundle_builder

    def preview(
        self,
        prepared: PreparedSpectralDataset,
        config: PostBaselineSmoothingConfig,
    ) -> PostBaselineSmoothingResult:
        """Run the authoritative core without creating a committed branch."""

        result = self._smoothing_core(prepared, config)
        if not isinstance(result, PostBaselineSmoothingResult):
            raise TypeError("smoothing core must return PostBaselineSmoothingResult")
        return result

    def apply(
        self,
        prepared: PreparedSpectralDataset,
        config: PostBaselineSmoothingConfig,
    ) -> tuple[PostBaselineSmoothingResult, PreparedSpectralDataset]:
        """Run the core once and create an explicit, inactive Prepared child."""

        if not isinstance(config, PostBaselineSmoothingConfig):
            raise TypeError("config must be a PostBaselineSmoothingConfig")
        if not config.enabled:
            raise ValueError("creating a smoothed scientific branch requires enabled=True")
        result = self.preview(prepared, config)
        child = self._prepared_adapter(result)
        if not isinstance(child, PreparedSpectralDataset):
            raise TypeError("prepared adapter must return PreparedSpectralDataset")
        return result, child

    def build_bundle(
        self,
        result: PostBaselineSmoothingResult,
        prepared: PreparedSpectralDataset,
    ) -> bytes:
        """Persist one committed child through the strict smoothing exporter."""

        if not isinstance(result, PostBaselineSmoothingResult):
            raise TypeError("result must be a PostBaselineSmoothingResult")
        if not isinstance(prepared, PreparedSpectralDataset):
            raise TypeError("prepared must be a PreparedSpectralDataset")
        bundle = self._bundle_builder(result, prepared)
        if not isinstance(bundle, bytes):
            raise TypeError("smoothing bundle builder must return bytes")
        return bundle


__all__ = ["PostBaselineSmoothingService", "SmoothingBundleBuilder"]
