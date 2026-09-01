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

SmoothingCore = Callable[
    [PreparedSpectralDataset, PostBaselineSmoothingConfig],
    PostBaselineSmoothingResult,
]
SmoothedPreparedAdapter = Callable[
    [PostBaselineSmoothingResult],
    PreparedSpectralDataset,
]


class PostBaselineSmoothingService:
    """Preview or commit one smoothing operation through the same core API.

    The service is deliberately independent of export and 2D-COS.  ``preview``
    returns only the immutable transformation result; ``apply`` additionally
    creates, but never activates, its Prepared child.
    """

    def __init__(
        self,
        *,
        smoothing_core: SmoothingCore = apply_post_baseline_smoothing,
        prepared_adapter: SmoothedPreparedAdapter = prepared_from_smoothed_result,
    ) -> None:
        if not callable(smoothing_core):
            raise TypeError("smoothing_core must be callable")
        if not callable(prepared_adapter):
            raise TypeError("prepared_adapter must be callable")
        self._smoothing_core = smoothing_core
        self._prepared_adapter = prepared_adapter

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


__all__ = ["PostBaselineSmoothingService"]
