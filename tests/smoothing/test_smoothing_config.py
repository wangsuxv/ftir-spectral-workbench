from __future__ import annotations

import json

import numpy as np
import pytest

from ftir_workbench.post_baseline_smoothing import PostBaselineSmoothingConfig


def test_defaults_are_disabled_and_complete_serialization_round_trips() -> None:
    config = PostBaselineSmoothingConfig()

    assert config.enabled is False
    assert config.method == "savgol"
    assert config.savgol_window_length == 7
    assert config.savgol_polyorder == 2
    assert config.gaussian_sigma_points == 1.0
    assert config.moving_average_window_length == 3
    assert config.median_window_length == 3
    assert config.scientific_dict() == {"enabled": False}
    assert PostBaselineSmoothingConfig.from_json(config.to_json()) == config
    assert json.loads(config.to_json())["gaussian_truncate"] == 4.0


@pytest.mark.parametrize(
    ("config", "parameters"),
    (
        (
            PostBaselineSmoothingConfig(
                enabled=True,
                method="savgol",
                savgol_window_length=9,
                savgol_polyorder=3,
                savgol_mode="mirror",
            ),
            {"window_length": 9, "polyorder": 3, "mode": "mirror"},
        ),
        (
            PostBaselineSmoothingConfig(
                enabled=True,
                method="gaussian",
                gaussian_sigma_points=1.5,
                gaussian_truncate=3.0,
                convolution_mode="nearest",
            ),
            {"sigma_points": 1.5, "truncate": 3.0, "mode": "nearest"},
        ),
        (
            PostBaselineSmoothingConfig(
                enabled=True,
                method="moving_average",
                moving_average_window_length=5,
                convolution_mode="mirror",
            ),
            {"window_length": 5, "mode": "mirror"},
        ),
        (
            PostBaselineSmoothingConfig(
                enabled=True,
                method="median",
                median_window_length=5,
                convolution_mode="reflect",
            ),
            {"window_length": 5, "mode": "reflect"},
        ),
    ),
)
def test_scientific_dict_contains_only_selected_method_parameters(
    config: PostBaselineSmoothingConfig,
    parameters: dict[str, int | float | str],
) -> None:
    scientific = config.scientific_dict()

    assert scientific == {
        "enabled": True,
        "method": config.method,
        "parameters": parameters,
        "axis": "wavenumber",
        "axis_index": 1,
        "uniformity_rtol": 1e-3,
        "nonuniform_axis_policy": "error",
    }
    assert PostBaselineSmoothingConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("updates", "error", "match"),
    (
        ({"enabled": 1}, TypeError, "enabled"),
        ({"method": "loess"}, ValueError, "method"),
        ({"savgol_window_length": True}, TypeError, "integer"),
        ({"savgol_window_length": np.bool_(True)}, TypeError, "integer"),
        ({"savgol_window_length": 2}, ValueError, "at least 3"),
        ({"savgol_window_length": 4}, ValueError, "odd"),
        ({"savgol_polyorder": -1}, ValueError, "non-negative"),
        ({"savgol_polyorder": 7}, ValueError, "less than"),
        ({"savgol_mode": "wrap"}, ValueError, "savgol_mode"),
        ({"gaussian_sigma_points": 0.0}, ValueError, "positive"),
        ({"gaussian_sigma_points": np.bool_(True)}, TypeError, "real number"),
        ({"gaussian_sigma_points": np.nan}, ValueError, "finite"),
        ({"gaussian_truncate": np.inf}, ValueError, "finite"),
        ({"moving_average_window_length": 6}, ValueError, "odd"),
        ({"median_window_length": 0}, ValueError, "at least 3"),
        ({"convolution_mode": "constant"}, ValueError, "convolution_mode"),
        ({"uniformity_rtol": -1.0}, ValueError, "non-negative"),
        ({"uniformity_rtol": np.inf}, ValueError, "finite"),
        ({"nonuniform_axis_policy": "resample"}, ValueError, "policy"),
    ),
)
def test_invalid_configuration_is_rejected(
    updates: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        PostBaselineSmoothingConfig(**updates)  # type: ignore[arg-type]


def test_choice_values_are_normalized_without_changing_meaning() -> None:
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="Moving-Average",  # type: ignore[arg-type]
        convolution_mode=" Nearest ",  # type: ignore[arg-type]
        nonuniform_axis_policy="Allow-Index-Space-With-Warning",  # type: ignore[arg-type]
    )

    assert config.method == "moving_average"
    assert config.convolution_mode == "nearest"
    assert config.nonuniform_axis_policy == "allow_index_space_with_warning"
