"""Naming preferences survive workspace round trips without scientific mutation."""
from __future__ import annotations

import json
from typing import Any

import pytest

from ftir_workbench.batch.export_naming import (
    get_naming_options,
    set_name_override,
    set_naming_policy,
)
from ftir_workbench.batch.models import BatchError
from ftir_workbench.batch.postprocessing import BRANCHES, get_branch
from ftir_workbench.batch.workspace import (
    load_batch_workspace,
    make_archive,
    read_verified_archive,
    save_batch_workspace,
)
from tests.postprocessing.helpers import confirmed_workspace
from tests.postprocessing.test_state_contracts import graph
from tests.postprocessing.test_workspace_migration import (
    assert_derived,
    assert_stage,
    forbid_calculations,
)

ARTIFACT_TYPE = "independent_baseline_workspace"


@pytest.mark.parametrize("policy", ["original", "display", "legacy_id"])
def test_naming_preferences_roundtrip_keeps_science_and_source_exact(
    policy: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="B.csv")
    graph(ws, a)
    original = ws.records[a]
    fingerprints = {branch: get_branch(ws, a, branch).fingerprint for branch in BRANCHES}
    forbid_calculations(monkeypatch)
    set_naming_policy(ws, policy)
    set_name_override(ws, a, "样品 α 90%RH")
    ws.states[a].display_preferences["zoom"] = [1650.0, 1150.0]
    ws.display_order = [b, a]
    ws.selected_spectrum_id = b
    restored = load_batch_workspace(save_batch_workspace(ws))
    assert get_naming_options(restored) == {"policy": policy, "overrides": {a: "样品 α 90%RH"}}
    assert restored.states[a].display_preferences["zoom"] == [1650.0, 1150.0]
    assert restored.display_order == [b, a] and restored.selected_spectrum_id == b
    assert ws.records[a] is original
    for sid in (a, b):
        assert restored.records[sid].display_name == ws.records[sid].display_name
        assert restored.records[sid].original_column_label == ws.records[sid].original_column_label
        assert restored.records[sid].scientific_input_sha256 == ws.records[sid].scientific_input_sha256
        source_id = ws.records[sid].source_id
        assert restored.sources[source_id].original_bytes == ws.sources[source_id].original_bytes
        for stage in ("coarse", "fine"):
            assert_stage(getattr(restored.states[sid], stage + "_snapshot"),
                         getattr(ws.states[sid], stage + "_snapshot"))
    for branch in BRANCHES:
        assert get_branch(restored, a, branch).fingerprint == fingerprints[branch]
        if branch != "baseline":
            assert_derived(get_branch(restored, a, branch), get_branch(ws, a, branch))


@pytest.mark.parametrize("namespace", [
    None, [], "original", {}, {"policy": "invalid"}, {"policy": True},
    {"policy": "original", "override": 3}, {"policy": "original", "override": []},
    {"policy": "original", "overrides": {}}, {"policy": "display", "unexpected": "value"},
])
@pytest.mark.parametrize("operation", ["save", "load"])
def test_malformed_naming_namespace_is_rejected_even_with_valid_manifest(
    namespace: Any, operation: str,
) -> None:
    ws, sid = confirmed_workspace()
    if operation == "save":
        ws.states[sid].display_preferences["export_naming"] = namespace
        with pytest.raises(BatchError, match="export naming") as caught:
            save_batch_workspace(ws)
    else:
        members = read_verified_archive(save_batch_workspace(ws), ARTIFACT_TYPE)
        data = json.loads(members["workspace.json"])
        data["states"][sid]["display_preferences"]["export_naming"] = namespace
        members["workspace.json"] = json.dumps(data).encode()
        with pytest.raises(BatchError, match="export naming") as caught:
            load_batch_workspace(make_archive(members, ARTIFACT_TYPE))
    assert caught.value.code == "WORKSPACE_INTEGRITY_FAILED"


@pytest.mark.parametrize("operation", ["save", "load"])
def test_conflicting_per_spectrum_workspace_policy_is_rejected(operation: str) -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="B.csv")
    set_naming_policy(ws, "original")
    if operation == "save":
        ws.states[b].display_preferences["export_naming"]["policy"] = "display"
        with pytest.raises(BatchError, match="conflicting"):
            save_batch_workspace(ws)
    else:
        members = read_verified_archive(save_batch_workspace(ws), ARTIFACT_TYPE)
        data = json.loads(members["workspace.json"])
        data["states"][a]["display_preferences"]["export_naming"]["policy"] = "legacy_id"
        members["workspace.json"] = json.dumps(data).encode()
        with pytest.raises(BatchError, match="conflicting"):
            load_batch_workspace(make_archive(members, ARTIFACT_TYPE))


def test_missing_namespace_and_newly_imported_record_share_explicit_policy() -> None:
    ws, a = confirmed_workspace()
    assert get_naming_options(ws) == {"policy": "original", "overrides": {}}
    set_naming_policy(ws, "display")
    set_name_override(ws, a, None)
    _, b = confirmed_workspace(workspace=ws, name="new.csv")
    assert "export_naming" not in ws.states[b].display_preferences
    restored = load_batch_workspace(save_batch_workspace(ws))
    assert get_naming_options(restored) == {"policy": "display", "overrides": {}}
    set_name_override(restored, b, "new override")
    assert restored.states[b].display_preferences["export_naming"]["policy"] == "display"


@pytest.mark.parametrize("override", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_override_is_rejected_before_editor_nan_to_null_repair(override: float) -> None:
    ws, sid = confirmed_workspace()
    ws.states[sid].display_preferences["export_naming"] = {
        "policy": "original", "override": override,
    }
    with pytest.raises(BatchError, match="export naming"):
        save_batch_workspace(ws)
