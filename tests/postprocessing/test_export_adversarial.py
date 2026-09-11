"""Rehashed untrusted postprocessing bundles must fail closed without replay."""

from __future__ import annotations

import copy
import csv
import io
from collections.abc import Callable

import numpy as np
import pytest

from ftir_workbench.batch.export import _table
from ftir_workbench.batch.fingerprints import json_fingerprint
from ftir_workbench.batch.postprocessing import array_hash
from ftir_workbench.batch.postprocessing_export import (
    ARTIFACT_TYPE,
    build_postprocessing_export,
    verify_postprocessing_export,
)
from ftir_workbench.batch.workspace import (
    _read_json,
    json_bytes,
    make_archive,
    read_verified_archive,
)
from tests.postprocessing.helpers import confirmed_workspace
from tests.postprocessing.test_state_contracts import apply_branch, graph


@pytest.fixture
def ordinary_payload() -> tuple[bytes, str]:
    ws, sid = confirmed_workspace()
    graph(ws, sid)
    return build_postprocessing_export(
        ws, [sid], branches=("baseline", "smoothed", "normalized_baseline", "normalized_smoothed"),
        include_wide=True,
    ).zip_bytes, sid


@pytest.fixture(autouse=True)
def no_verification_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("recompute=False verifier must not replay baseline, S or N")

    monkeypatch.setattr("ftir_workbench.batch.postprocessing_export.smooth_spectral_arrays", forbidden)
    monkeypatch.setattr("ftir_workbench.batch.postprocessing_export.normalize_spectral_arrays", forbidden)


def rehash_bundle(payload: bytes, mutate: Callable[[dict, dict, dict[str, dict], dict[str, bytes]], None]) -> bytes:
    """Regenerate transport/node hashes and report copies after malicious edits.

    A rejection cannot be attributed merely to stale ZIP or node checksums.
    Scientific request/output hashes are changed only by the specific attack.
    """
    members = read_verified_archive(payload, ARTIFACT_TYPE)
    metadata = _read_json(members["batch_metadata.json"])
    index = _read_json(members["branch_index.json"])
    nodes = {entry["recipe_json"]: _read_json(members[entry["recipe_json"]]) for entry in index["nodes"]}
    mutate(metadata, index, nodes, members)
    for path, node in nodes.items():
        node["node_integrity_sha256"] = json_fingerprint(
            {key: value for key, value in node.items() if key != "node_integrity_sha256"}
        )
        members[path] = json_bytes(node)
    headers = sorted({key for row in metadata["report"] for key in row})
    members["processing_report.csv"] = _table(headers, [[
        json_bytes(row.get(key)).decode() if isinstance(row.get(key), (list, dict)) else row.get(key, "")
        for key in headers] for row in metadata["report"]])
    members["batch_metadata.json"] = json_bytes(metadata)
    members["branch_index.json"] = json_bytes(index)
    return make_archive(members, ARTIFACT_TYPE)


def node_for(nodes: dict[str, dict], branch: str) -> dict:
    return next(node for node in nodes.values() if node["branch"] == branch)


def test_untouched_bundle_verifies_without_any_replay(ordinary_payload: tuple[bytes, str]) -> None:
    assert verify_postprocessing_export(ordinary_payload[0], recompute=False)


@pytest.mark.parametrize("attack", [
    "source_id", "spectrum_id", "workspace_id", "source_column", "parent_branch", "parent_hash",
    "quantity", "purpose", "coarse_parent", "fine_not_decided", "duplicated_node", "missing_parent",
    "wrong_inclusion", "unsafe_recipe_path", "aliased_recipe", "unknown_branch", "extra_member",
    "duplicate_requested_branch", "lost_report_row", "invented_report_row", "false_2d", "wide_axes_claim",
    "baseline_unit_record", "invalid_original_direction",
])
def test_rehashed_identity_graph_quantity_and_report_conflicts_return_false(
    ordinary_payload: tuple[bytes, str], attack: str,
) -> None:
    payload, sid = ordinary_payload

    def corrupt(metadata: dict, index: dict, nodes: dict[str, dict], members: dict[str, bytes]) -> None:
        baseline = node_for(nodes, "baseline")
        normalized = node_for(nodes, "normalized_smoothed")
        if attack == "source_id":
            normalized["source_id"] = normalized["source"]["source_id"] = "other-source"
        elif attack == "spectrum_id":
            normalized["spectrum_id"] = normalized["source"]["spectrum_id"] = "other-spectrum"
        elif attack == "workspace_id":
            normalized["workspace_id"] = "other-workspace"
        elif attack == "source_column":
            source = normalized["source"]
            source["source_column_index"] += 1
            source["source_selection_sha256"] = json_fingerprint({
                "source_sha256": source["source_sha256"], "column_index": source["source_column_index"],
                "import_options": source["import_options"],
            })
        elif attack == "parent_branch":
            normalized["parent_branch"] = "normalized_baseline"
        elif attack == "parent_hash":
            normalized["parent_fingerprint"] = baseline["fingerprint"]
        elif attack == "quantity":
            normalized["quantity"] = "absorbance"
        elif attack == "purpose":
            normalized["purpose"] = "display_only"
        elif attack == "coarse_parent":
            baseline["stage_request"]["parent_coarse_fingerprint"] = "0" * 64
        elif attack == "fine_not_decided":
            baseline["fine_decision"] = "not_decided"
        elif attack == "duplicated_node":
            index["nodes"].append(copy.deepcopy(index["nodes"][0]))
        elif attack == "missing_parent":
            index["nodes"] = [row for row in index["nodes"] if row["branch"] != "smoothed"]
        elif attack == "wrong_inclusion":
            index["nodes"][0]["included_as"] = "parent"
        elif attack == "unsafe_recipe_path":
            path = next(path for path, node in nodes.items() if node is normalized)
            del nodes[path]
            normalized["recipe_json"] = "../outside.json"
            nodes["../outside.json"] = normalized
            next(row for row in index["nodes"] if row["branch"] == "normalized_smoothed")["recipe_json"] = "../outside.json"
        elif attack == "aliased_recipe":
            index["nodes"][-1]["recipe_json"] = index["nodes"][0]["recipe_json"]
        elif attack == "unknown_branch":
            index["nodes"][-1]["branch"] = "normalized_twice"
        elif attack == "extra_member":
            members["unrequested.npy"] = b"no arrays or executable objects belong in this CSV bundle"
        elif attack == "duplicate_requested_branch":
            metadata["requested_branches"].append("smoothed")
        elif attack == "lost_report_row":
            metadata["report"].pop()
        elif attack == "invented_report_row":
            metadata["report"].append({"spectrum_id": sid, "branch": "normalized_twice", "status": "exported"})
        elif attack == "false_2d":
            normalized["is_2d_ready"] = True
        elif attack == "wide_axes_claim":
            metadata["wide_available"] = False
        elif attack == "baseline_unit_record":
            baseline["unit_conversion"]["input_unit"] = "percent_transmittance"
        else:
            for node in nodes.values():
                node["source"]["original_axis_direction"] = "not-an-axis-direction"
            for row in metadata["report"]:
                row["original_axis_direction"] = "not-an-axis-direction"
    assert verify_postprocessing_export(rehash_bundle(payload, corrupt), recompute=False) is False


@pytest.mark.parametrize("attack", [
    "nan", "infinity", "expression", "duplicate_axis", "wrong_column", "huge_header", "object_bytes",
])
def test_malicious_csv_payloads_return_false_without_uncaught_parser_errors(
    ordinary_payload: tuple[bytes, str], attack: str,
) -> None:
    def corrupt(metadata: dict, index: dict, nodes: dict[str, dict], members: dict[str, bytes]) -> None:
        node = node_for(nodes, "normalized_baseline")
        path = node["spectra_csv"]
        rows = list(csv.reader(io.StringIO(members[path].decode())))
        if attack == "huge_header":
            members[path] = ("x" * (csv.field_size_limit() + 1) + ",normalized_intensity\n0,0\n1,1\n").encode()
            return
        if attack == "object_bytes":
            output = io.BytesIO()
            np.save(output, np.array([{"must_not_execute": True}], dtype=object), allow_pickle=True)
            members[path] = output.getvalue()
            return
        if attack in {"nan", "infinity", "expression"}:
            rows[1][1] = {"nan": "nan", "infinity": "inf", "expression": "=1+1"}[attack]
        elif attack == "duplicate_axis":
            rows[2][0] = rows[1][0]
        else:
            rows[0][1] = "corrected_absorbance"
        output = io.StringIO(newline="")
        csv.writer(output, lineterminator="\n").writerows(rows)
        members[path] = output.getvalue().encode()
    assert verify_postprocessing_export(rehash_bundle(ordinary_payload[0], corrupt), recompute=False) is False


def test_rehashed_extreme_minmax_parent_returns_false_instead_of_floating_point_exception() -> None:
    ws, sid = confirmed_workspace(np.arange(3.0), np.array([-1.0, 1.0, 3.0]))
    apply_branch(ws, sid, "normalized_baseline", {"enabled": True, "method": "minmax_display"})
    payload = build_postprocessing_export(ws, [sid], branches=("normalized_baseline",)).zip_bytes

    def corrupt(metadata: dict, index: dict, nodes: dict[str, dict], members: dict[str, bytes]) -> None:
        parent = node_for(nodes, "baseline")
        child = node_for(nodes, "normalized_baseline")
        values = np.array([[-1e308, 0.0, 1e308]])
        members[parent["spectra_csv"]] = _table(
            ["wavenumber_cm-1", "corrected_absorbance"],
            list(zip(np.arange(3.0), values[0], strict=True)),
        )
        parent["y_sha256"] = array_hash(values)
        child["request_payload"]["parent_y_sha256"] = parent["y_sha256"]
        child["request_fingerprint"] = json_fingerprint(child["request_payload"])
    assert verify_postprocessing_export(rehash_bundle(payload, corrupt), recompute=False) is False
