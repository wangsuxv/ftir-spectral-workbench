from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ftir_workbench.cli import _smoothing_config, build_parser, main
from ftir_workbench.export import (
    load_prepared,
    serialize_prepared,
    verify_twodcos_bundle,
)
from ftir_workbench.smoothing_export import verify_smoothing_bundle
from tests.smoothing._helpers import make_prepared

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_prepared_pair(tmp_path: Path, *, nonuniform: bool = False) -> tuple[Path, Path]:
    if nonuniform:
        steps = np.linspace(18.0, 22.0, 40)
        axis = np.concatenate(([1800.0], 1800.0 - np.cumsum(steps)))
        prepared = make_prepared(wavenumber=axis)
    else:
        prepared = make_prepared()
    artifact = serialize_prepared(prepared)
    csv_path = tmp_path / artifact.csv_name
    metadata_path = tmp_path / artifact.metadata_name
    csv_path.write_bytes(artifact.csv_bytes)
    metadata_path.write_bytes(artifact.metadata_bytes)
    return csv_path, metadata_path


@pytest.mark.parametrize(
    ("method", "method_args", "expected_parameters"),
    [
        (
            "savgol",
            ["--window-length", "7", "--polyorder", "2", "--mode", "mirror"],
            {"window_length": 7, "polyorder": 2, "mode": "mirror"},
        ),
        (
            "gaussian",
            ["--sigma-points", "1.25", "--truncate", "3", "--mode", "nearest"],
            {"sigma_points": 1.25, "truncate": 3.0, "mode": "nearest"},
        ),
        (
            "moving_average",
            ["--window-length", "5", "--mode", "reflect"],
            {"window_length": 5, "mode": "reflect"},
        ),
        (
            "median",
            ["--window-length", "3", "--mode", "mirror"],
            {"window_length": 3, "mode": "mirror"},
        ),
    ],
)
def test_cli_smooth_supports_each_method_from_prepared_pair(
    method: str,
    method_args: list[str],
    expected_parameters: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path, metadata_path = _write_prepared_pair(tmp_path)
    output = tmp_path / f"{method}-smoothing.zip"

    exit_code = main(
        [
            "smooth",
            str(csv_path),
            "--metadata",
            str(metadata_path),
            "--method",
            method,
            *method_args,
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["state"] == "smoothing_completed"
    assert summary["method"] == method
    assert summary["effective_parameters"] == expected_parameters
    assert summary["manifest_verified"] is True
    assert summary["parent_prepared_data_sha256"] != summary["prepared_data_sha256"]
    assert output.is_file()
    assert verify_smoothing_bundle(output)
    child = load_prepared(output)
    assert child.prepared_data_sha256 == summary["prepared_data_sha256"]
    assert child.baseline_recipe["prepared_data_contract"]["branch_kind"] == (
        "post_baseline_smoothing"
    )


def test_smooth_command_is_always_enabled_and_parameters_are_method_scoped(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        ["smooth", "prepared.csv", "--method", "savgol"]
    )
    config = _smoothing_config(arguments)

    assert config.enabled is True
    smooth_parser = next(
        action
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    ).choices["smooth"]
    option_strings = {
        option
        for action in smooth_parser._actions
        for option in action.option_strings
    }
    assert "--enabled" not in option_strings
    assert "--disabled" not in option_strings

    assert (
        main(
            [
                "smooth",
                "missing-prepared.csv",
                "--method",
                "gaussian",
                "--window-length",
                "5",
            ]
        )
        == 2
    )
    assert "--window-length is not valid with --method gaussian" in (
        capsys.readouterr().err
    )

    assert (
        main(
            [
                "smooth",
                "missing-prepared.csv",
                "--method",
                "savgol",
                "--mode",
                "reflect",
            ]
        )
        == 2
    )
    assert "--mode reflect is not valid with --method savgol" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect", "raw.csv"],
        ["baseline", "raw.csv"],
        ["twodcos", "prepared.csv", "--range", "1800:900"],
        ["demo"],
        ["verify", "bundle.zip"],
    ],
)
def test_existing_commands_do_not_receive_smooth_only_arguments(argv: list[str]) -> None:
    arguments = build_parser().parse_args(argv)

    for name in (
        "window_length",
        "polyorder",
        "sigma_points",
        "truncate",
        "uniformity_rtol",
        "nonuniform_axis_policy",
    ):
        assert not hasattr(arguments, name)


def test_cli_smooth_nonuniform_axis_requires_explicit_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path, metadata_path = _write_prepared_pair(tmp_path, nonuniform=True)
    rejected = tmp_path / "rejected.zip"

    assert (
        main(
            [
                "smooth",
                str(csv_path),
                "--metadata",
                str(metadata_path),
                "--method",
                "gaussian",
                "--output",
                str(rejected),
            ]
        )
        == 2
    )
    assert "not approximately uniform" in capsys.readouterr().err
    assert not rejected.exists()

    allowed = tmp_path / "allowed.zip"
    assert (
        main(
            [
                "smooth",
                str(csv_path),
                "--metadata",
                str(metadata_path),
                "--method",
                "gaussian",
                "--nonuniform-axis-policy",
                "allow_index_space_with_warning",
                "--output",
                str(allowed),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert any("Index-space smoothing" in warning for warning in summary["warnings"])
    assert verify_smoothing_bundle(allowed)


def test_cli_baseline_smooth_verify_twodcos_chain_and_chaining_rejection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = PROJECT_ROOT / "examples" / "baseline" / "example_absorbance.csv"
    baseline = tmp_path / "baseline.zip"
    assert (
        main(
            [
                "baseline",
                str(source),
                "--unit",
                "absorbance",
                "--baseline-high",
                "1800",
                "--baseline-low",
                "900",
                "--coarse-method",
                "none",
                "--fine-method",
                "none",
                "--series-mode",
                "independent_locked",
                "--output",
                str(baseline),
            ]
        )
        == 0
    )
    capsys.readouterr()

    smoothing = tmp_path / "post_baseline_smoothing_run.zip"
    assert (
        main(
            [
                "smooth",
                str(baseline),
                "--method",
                "gaussian",
                "--sigma-points",
                "1",
                "--nonuniform-axis-policy",
                "allow_index_space_with_warning",
                "--output",
                str(smoothing),
            ]
        )
        == 0
    )
    smoothing_summary = json.loads(capsys.readouterr().out)
    assert smoothing_summary["manifest_verified"] is True

    assert main(["verify", str(smoothing)]) == 0
    verify_summary = json.loads(capsys.readouterr().out)
    assert verify_summary["bundle_type"] == "smoothing"
    assert verify_summary["verified"] is True

    twodcos = tmp_path / "twodcos.zip"
    assert (
        main(
            [
                "twodcos",
                str(smoothing),
                "--range",
                "1800:900:full",
                "--output",
                str(twodcos),
            ]
        )
        == 0
    )
    twodcos_summary = json.loads(capsys.readouterr().out)
    assert twodcos_summary["state"] == "twodcos_completed"
    assert verify_twodcos_bundle(twodcos)

    chained = tmp_path / "chained.zip"
    assert (
        main(
            [
                "smooth",
                str(smoothing),
                "--method",
                "gaussian",
                "--output",
                str(chained),
            ]
        )
        == 2
    )
    assert "already a post-baseline smoothing branch" in capsys.readouterr().err
    assert not chained.exists()


def test_twodcos_help_mentions_smoothing_bundle_without_new_science_arguments() -> None:
    parser = build_parser()
    command_action = next(
        action
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    )
    twodcos_parser = command_action.choices["twodcos"]

    assert "smoothing bundle" in twodcos_parser.format_help()
    twodcos_options = {
        option
        for action in twodcos_parser._actions
        for option in action.option_strings
    }
    assert "--method" not in twodcos_options
    assert "--window-length" not in twodcos_options
    assert "--sigma-points" not in twodcos_options
