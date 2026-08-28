from __future__ import annotations

import io
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from ftir_baseline.units import (
    fraction_transmittance_to_absorbance,
    percent_transmittance_to_absorbance,
)
from ftir_workbench.display_units import (
    DisplayConversionResult,
    absorbance_to_fraction_transmittance,
    absorbance_to_percent_transmittance,
    convert_absorbance_for_display,
    derived_transmittance_csv_bytes,
    derived_transmittance_filename,
)


def test_fraction_transmittance_reference_values_and_round_trip() -> None:
    absorbance = np.array([-0.25, 0.0, 0.5, 1.0, 2.0], dtype=np.float64)

    converted = absorbance_to_fraction_transmittance(absorbance)

    np.testing.assert_allclose(converted.values, 10.0 ** (-absorbance), rtol=1e-15)
    np.testing.assert_allclose(
        fraction_transmittance_to_absorbance(converted.values),
        absorbance,
        rtol=1e-14,
        atol=1e-15,
    )
    assert converted.input_unit == "absorbance"
    assert converted.output_unit == "fraction_transmittance"
    assert converted.formula == "T = 10 ** (-A)"


def test_percent_transmittance_reference_values_and_round_trip() -> None:
    absorbance = np.array([-0.25, 0.0, 0.5, 1.0, 2.0], dtype=np.float64)

    converted = absorbance_to_percent_transmittance(absorbance)

    np.testing.assert_allclose(converted.values, 100.0 * 10.0 ** (-absorbance), rtol=1e-15)
    np.testing.assert_allclose(
        percent_transmittance_to_absorbance(converted.values),
        absorbance,
        rtol=1e-14,
        atol=1e-15,
    )
    assert converted.output_unit == "percent_transmittance"
    assert converted.formula == "%T = 100 * 10 ** (-A)"


def test_negative_absorbance_is_not_clipped_and_is_reported() -> None:
    percent = absorbance_to_percent_transmittance([-0.5])
    fraction = absorbance_to_fraction_transmittance([-0.5])

    assert percent.values[0] > 100.0
    assert fraction.values[0] > 1.0
    assert percent.warnings
    assert fraction.warnings
    assert "not clipped" in percent.warnings[0]


def test_identity_is_detached_float64_immutable_copy() -> None:
    source = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    original = source.copy()

    converted = convert_absorbance_for_display(source, "absorbance")

    np.testing.assert_allclose(converted.values, source)
    np.testing.assert_array_equal(source, original)
    assert converted.values.dtype == np.float64
    assert not np.shares_memory(converted.values, source)
    assert not converted.values.flags.writeable
    with pytest.raises(ValueError):
        converted.values.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        converted.output_unit = "percent_transmittance"  # type: ignore[misc]


def test_transmittance_output_is_detached_and_preserves_input_shape() -> None:
    source = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 10.0
    original = source.copy()

    converted = convert_absorbance_for_display(source, "fraction_transmittance")

    assert converted.values.shape == source.shape
    assert converted.values.dtype == np.float64
    assert not np.shares_memory(converted.values, source)
    assert not converted.values.flags.writeable
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_absorbance_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        convert_absorbance_for_display([0.0, bad], "percent_transmittance")


@pytest.mark.parametrize(
    "values",
    [
        [1.0 + 2.0j],
        np.array([1.0 + 0.0j], dtype=np.complex128),
    ],
)
def test_complex_absorbance_is_rejected(values: object) -> None:
    with pytest.raises(TypeError, match="complex"):
        convert_absorbance_for_display(values, "fraction_transmittance")  # type: ignore[arg-type]


def test_fraction_transmittance_overflow_is_not_silenced() -> None:
    with pytest.raises(FloatingPointError, match="overflow"):
        absorbance_to_fraction_transmittance([-400.0])


def test_percent_transmittance_multiplication_overflow_is_not_silenced() -> None:
    with pytest.raises(FloatingPointError, match="overflow"):
        absorbance_to_percent_transmittance([-307.0])


def test_dispatch_rejects_unknown_display_unit() -> None:
    with pytest.raises(ValueError, match="unsupported display intensity unit"):
        convert_absorbance_for_display([0.0], "radiance")  # type: ignore[arg-type]


def test_result_constructor_enforces_units_and_immutable_values() -> None:
    with pytest.raises(ValueError, match="input_unit"):
        DisplayConversionResult(
            values=np.array([1.0]),
            input_unit="percent_transmittance",  # type: ignore[arg-type]
            output_unit="absorbance",
            formula="invalid",
            warnings=(),
        )

    source = np.array([0.1, 0.2])
    result = DisplayConversionResult(
        values=source,
        input_unit="absorbance",
        output_unit="absorbance",
        formula="A = absorbance (identity copy)",
        warnings=[],  # type: ignore[arg-type]
    )
    source[0] = 99.0
    np.testing.assert_array_equal(result.values, [0.1, 0.2])
    assert result.warnings == ()


@pytest.mark.parametrize(
    ("unit", "filename", "expected"),
    (
        (
            "fraction_transmittance",
            "derived_fraction_transmittance_from_corrected_absorbance.csv",
            np.array([[1.0, 0.1], [10.0, 0.01]]),
        ),
        (
            "percent_transmittance",
            "derived_percent_transmittance_from_corrected_absorbance.csv",
            np.array([[100.0, 10.0], [1000.0, 1.0]]),
        ),
    ),
)
def test_derived_csv_is_labelled_independent_non_instrument_export(
    unit: str,
    filename: str,
    expected: np.ndarray,
) -> None:
    axis = np.array([1800.0, 1700.0])
    absorbance = np.array([[0.0, 1.0], [-1.0, 2.0]])
    original = absorbance.copy()

    payload = derived_transmittance_csv_bytes(
        axis,
        absorbance,
        ("first", "second"),
        output_unit=unit,  # type: ignore[arg-type]
    )
    text = payload.decode("utf-8")
    table = pd.read_csv(io.BytesIO(payload), comment="#")

    assert derived_transmittance_filename(unit) == filename  # type: ignore[arg-type]
    assert "mathematical representation derived from baseline-corrected absorbance" in text
    assert "not the original instrument transmittance" in text
    assert "values were not clipped" in text
    assert table.columns.tolist() == ["Wavenumber", "first", "second"]
    np.testing.assert_array_equal(table["Wavenumber"].to_numpy(), axis)
    np.testing.assert_allclose(table[["first", "second"]].to_numpy().T, expected)
    np.testing.assert_array_equal(absorbance, original)


def test_derived_csv_rejects_shape_and_unit_mismatches() -> None:
    with pytest.raises(ValueError, match="shape"):
        derived_transmittance_csv_bytes(
            [1800.0, 1700.0],
            [[0.0]],
            ("only",),
            output_unit="fraction_transmittance",
        )
    with pytest.raises(ValueError, match="derived export unit"):
        derived_transmittance_filename("absorbance")  # type: ignore[arg-type]
