from __future__ import annotations

import numpy as np
import pytest

from ftir_baseline.models import SpectrumSet
from ftir_baseline.units import (
    InvalidTransmittanceError,
    convert_to_absorbance,
    fraction_transmittance_to_absorbance,
    percent_transmittance_to_absorbance,
    to_absorbance,
)
from ftir_baseline.validation import SpectrumValidationError


@pytest.mark.parametrize(
    ("values", "unit", "expected"),
    [
        ([100.0], "percent_transmittance", [0.0]),
        ([10.0], "percent_transmittance", [1.0]),
        ([0.1], "fraction_transmittance", [1.0]),
    ],
)
def test_conversion_reference_values(values: list[float], unit: str, expected: list[float]) -> None:
    actual = to_absorbance(values, unit)  # type: ignore[arg-type]
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert actual.dtype == np.float64


def test_named_conversion_helpers() -> None:
    np.testing.assert_allclose(percent_transmittance_to_absorbance([100, 10]), [0, 1])
    np.testing.assert_allclose(fraction_transmittance_to_absorbance([1, 0.1]), [0, 1])


def test_percent_formula_does_not_underflow_tiny_positive_float() -> None:
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    converted = percent_transmittance_to_absorbance([tiny])
    assert np.isfinite(converted[0])


@pytest.mark.parametrize("bad", [[0.0], [-0.01], [20.0, 0.0, -1.0]])
def test_nonpositive_transmittance_stops_with_positions(bad: list[float]) -> None:
    with pytest.raises(InvalidTransmittanceError, match=r"<= 0 at positions"):
        to_absorbance(bad, "percent_transmittance")


@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [-np.inf]])
def test_nonfinite_intensity_always_stops(bad: list[float]) -> None:
    with pytest.raises(SpectrumValidationError, match="non-finite"):
        convert_to_absorbance(bad, "fraction_transmittance", transmittance_floor=0.1)


def test_explicit_floor_returns_complete_repair_record() -> None:
    values = np.array([[10.0, 0.0], [-2.0, 0.05]])
    result = convert_to_absorbance(values, "percent_transmittance", transmittance_floor=0.1)

    np.testing.assert_allclose(result.absorbance, [[1.0, 3.0], [3.0, 3.0]])
    assert result.record.transmittance_floor == 0.1
    assert result.record.repaired_count == 3
    assert result.record.repaired_indices == ((0, 1), (1, 0), (1, 1))
    assert result.record.repaired is True
    assert result.record.to_dict()["formula"].startswith("A = -log10")


def test_floor_cannot_be_used_while_discarding_record() -> None:
    with pytest.raises(ValueError, match="return_record=True"):
        to_absorbance([0.0], "percent_transmittance", transmittance_floor=0.1)


def test_to_absorbance_can_return_record() -> None:
    absorbance, record = to_absorbance(
        [0.0, 10.0],
        "percent_transmittance",
        transmittance_floor=0.1,
        return_record=True,
    )
    np.testing.assert_allclose(absorbance, [3.0, 1.0])
    assert record.repaired_count == 1


def test_conversion_never_modifies_input() -> None:
    values = np.array([100.0, 10.0], dtype=np.float32)
    original = values.copy()
    converted = percent_transmittance_to_absorbance(values)

    np.testing.assert_array_equal(values, original)
    assert converted.dtype == np.float64
    assert not np.shares_memory(values, converted)


def test_absorbance_identity_is_a_detached_float64_copy() -> None:
    values = np.array([0.2, 0.4], dtype=np.float32)
    result = convert_to_absorbance(values, "absorbance")
    np.testing.assert_allclose(result.absorbance, values)
    assert result.absorbance.dtype == np.float64
    assert not np.shares_memory(result.absorbance, values)


def test_spectrum_set_is_deeply_immutable_and_float64() -> None:
    x = np.array([1000, 999, 998], dtype=np.int32)
    y = np.array([[1, 2, 3]], dtype=np.float32)
    metadata = {"nested": {"items": [1, 2]}}
    data = SpectrumSet(
        wavenumber=x,
        perturbation=np.array([0], dtype=np.int16),
        perturbation_labels=("0 min",),
        spectra=y,
        intensity_unit="absorbance",
        source_name="memory",
        metadata=metadata,
    )

    x[0] = 1
    y[0, 0] = 99
    metadata["nested"]["items"].append(3)
    assert data.wavenumber[0] == 1000
    assert data.spectra[0, 0] == 1
    assert data.metadata["nested"]["items"] == (1, 2)
    assert data.wavenumber.dtype == data.spectra.dtype == np.float64

    with pytest.raises(ValueError, match="WRITEABLE"):
        data.spectra.flags.writeable = True
    with pytest.raises(TypeError):
        data.metadata["new"] = "not allowed"
    with pytest.raises(TypeError):
        data.metadata["nested"]["new"] = "not allowed"


def test_spectrum_set_rejects_shape_and_label_mismatches() -> None:
    with pytest.raises(SpectrumValidationError, match="point count"):
        SpectrumSet(
            wavenumber=np.array([1.0, 2.0]),
            perturbation=np.array([0.0]),
            perturbation_labels=("0",),
            spectra=np.array([[1.0, 2.0, 3.0]]),
            intensity_unit="absorbance",
            source_name="bad",
            metadata={},
        )


def test_recipe_round_trip_uses_public_lambda_alias() -> None:
    pydantic = pytest.importorskip("pydantic")
    assert pydantic is not None
    from ftir_baseline.config import PipelineConfig

    recipe = PipelineConfig()
    payload = recipe.to_json()
    restored = PipelineConfig.from_json(payload)

    assert restored == recipe
    assert '"lambda"' in payload
    assert '"lam"' not in payload
