from collections import Counter
from pathlib import Path
import json
from scripts.verify_v031_browser_artifacts import (
    inspect_export, inspect_workspace, compare_export_workspace,
    assert_equal, require,
)
from ftir_workbench.batch.postprocessing import branch_state, get_branch
directory = Path("outputs/validation-private/v031/browser")
ws, ws_report = inspect_workspace(directory / "workspace_v031.zip", directory / "inputs")
old, _ = inspect_workspace(directory / "workspace_before.zip", directory / "inputs")
counts = Counter()
assert_equal(old.states, ws.states, "all_original_baseline_state", counts)
report = {"status": "pass", "workspace": ws_report,
          "unchanged_baseline_arrays": counts["arrays"], "exports": {}}
sid_a = next(sid for sid, rec in ws.records.items() if rec.display_name == "synthetic_A")
for branch in ("smoothed", "normalized_smoothed"):
    snap = get_branch(ws, sid_a, branch)
    require(snap.versions["ftir-spectral-workbench"] == "0.3.1", "A branch was not computed under final v0.3.1")
    require(snap.fingerprint != get_branch(old, sid_a, branch).fingerprint, "A final branch was not updated")
assert_equal(branch_state(old, sid_a, "normalized_baseline").committed,
             branch_state(ws, sid_a, "normalized_baseline").committed,
             "A_NB_unchanged_after_new_S_and_NS", counts)
for sid in ws.records:
    if sid != sid_a:
        assert_equal(old.postprocessing[sid], ws.postprocessing[sid], "unrelated_"+sid, counts)
for filename in ("v031_smoothed.zip", "v031_normalized_smoothed.zip"):
    bundle, info = inspect_export(directory / filename)
    require(info["exporter_versions"]["ftir-spectral-workbench"] == "0.3.1", "Final exporter version mismatch")
    arrays = compare_export_workspace(bundle, ws)
    report["exports"][filename] = {"integrity": info, "confirmed_arrays": arrays}
report["A_NB_and_unrelated_B_C_unchanged"] = True
report["final_A_S_and_NS_computation_versions"] = "0.3.1"
report["scope"] = "Actual final-version browser downloads; explicit verifier replays only S/N, never baseline."
out = Path("artifacts/validation/v0.3.1/phase5/browser_final_artifact_audit.json")
out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
