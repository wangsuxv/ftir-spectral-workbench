"""Pure export names, independent of scientific execution and snapshot identity.

Names are allocated across the whole workspace before selecting result nodes.
The fixed base budget reserves the longest supported suffix, so every branch
uses the same base, including when it is exported through another entry point.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from .models import BatchError, BatchWorkspace

NAMING_POLICY_VERSION = "1.0"
POLICIES = ("original", "display", "legacy_id")
MAX_FILENAME_BYTES = 200
# Includes the longest currently supported derived T/%T filename and prefix.
BASE_BUDGET = 100
Naming = Mapping[str, Any] | str | None
_NORMALIZATION = {
    "maximum": "norm-max", "internal_peak_height": "norm-peak-window",
    "internal_peak_area": "norm-area-window", "area": "norm-area",
    "vector": "norm-l2", "minmax_display": "minmax_scaled_intensity_display_only",
}
_SMOOTHING = {"savgol", "gaussian", "moving_average", "median"}
_DECISION_KEYS = (
    "spectrum_id", "source_id", "source_column_index", "original_filename",
    "original_column_label", "display_name", "source_sha256", "source_layout",
    "layout_evidence", "policy", "policy_version", "override", "name_basis",
    "raw_base_name", "base_name", "notes", "collision_token", "collision_reason",
    "filename_qualifier",
)


def _error(message: str) -> BatchError:
    return BatchError("INVALID_EXPORT_NAMING", message)


def validate_naming_preferences(preferences: Mapping[str, Any]) -> None:
    if "export_naming" not in preferences:
        return
    value = preferences["export_naming"]
    if not isinstance(value, dict) or not {"policy"} <= value.keys() <= {"policy", "override"}:
        raise _error("export_naming must contain policy and optional override")
    if not isinstance(value["policy"], str) or value["policy"] not in POLICIES:
        raise _error("unknown naming policy")
    if value.get("override") is not None and not isinstance(value["override"], str):
        raise _error("export name override must be text or null")


def validate_workspace_naming(preferences: Sequence[Mapping[str, Any]]) -> None:
    for item in preferences:
        validate_naming_preferences(item)
    policies = {item["export_naming"]["policy"] for item in preferences if "export_naming" in item}
    if len(policies) > 1:
        raise _error("workspace contains conflicting export naming policies")


def get_naming_options(workspace: BatchWorkspace, naming: Naming = None) -> dict[str, Any]:
    preferences = [state.display_preferences for state in workspace.states.values()]
    validate_workspace_naming(preferences)
    explicit = [item["export_naming"] for item in preferences if "export_naming" in item]
    policy = explicit[0]["policy"] if explicit else "original"
    overrides = {sid: state.display_preferences["export_naming"]["override"]
                 for sid, state in workspace.states.items()
                 if state.display_preferences.get("export_naming", {}).get("override") is not None}
    if isinstance(naming, str):
        policy = naming
    elif naming is not None:
        if not isinstance(naming, Mapping) or not naming.keys() <= {"policy", "overrides"}:
            raise _error("naming options require policy and/or overrides")
        policy = naming.get("policy", policy)
        supplied = naming.get("overrides", {})
        if not isinstance(supplied, Mapping):
            raise _error("overrides must be a spectrum_id to text mapping")
        for sid, value in supplied.items():
            if not isinstance(sid, str) or sid not in workspace.records:
                raise _error("override refers to an unknown spectrum")
            if value is None:
                overrides.pop(sid, None)
            elif isinstance(value, str):
                overrides[sid] = value
            else:
                raise _error("override must be text or null")
    if not isinstance(policy, str) or policy not in POLICIES:
        raise _error("unknown naming policy")
    return {"policy": policy, "overrides": overrides}


def set_naming_policy(workspace: BatchWorkspace, policy: str) -> None:
    get_naming_options(workspace, policy)
    for state in workspace.states.values():
        previous = state.display_preferences.get("export_naming", {})
        state.display_preferences["export_naming"] = {**previous, "policy": policy}


def set_name_override(workspace: BatchWorkspace, spectrum_id: str, value: str | None) -> None:
    options = get_naming_options(workspace)
    if spectrum_id not in workspace.records or (value is not None and not isinstance(value, str)):
        raise _error("override requires a known spectrum and text or null")
    workspace.states[spectrum_id].display_preferences["export_naming"] = {
        "policy": options["policy"], "override": value,
    }


def filename_stem(value: str) -> str:
    name = re.split(r"[/\\]", value)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def _clean(value: str, fallback: str) -> tuple[str, list[str]]:
    normalized = unicodedata.normalize("NFC", value)
    cleaned = "".join(char for char in normalized if unicodedata.category(char) not in {"Cc", "Cf", "Cs"})
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", cleaned).rstrip(" .")
    notes = []
    if normalized != value:
        notes.append("Unicode normalized to NFC")
    if cleaned != normalized:
        notes.append("unsafe characters removed/replaced or trailing dots/spaces removed")
    if not cleaned.strip() or cleaned in {".", ".."}:
        cleaned = fallback
        notes.append("unusable name: spectrum/column fallback")
    if re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", cleaned, re.IGNORECASE):
        cleaned = "_" + cleaned
        notes.append("Windows reserved name prefixed")
    return cleaned, notes


def _truncate(value: str, budget: int) -> str:
    return value.encode("utf-8")[:budget].decode("utf-8", errors="ignore").rstrip(" .")


def _key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _identity_token(spectrum_id: str, source_id: str) -> str:
    return hashlib.sha256(json.dumps([spectrum_id, source_id], ensure_ascii=True).encode()).hexdigest()


def _layout(evidence: Mapping[str, Any]) -> str:
    probe = evidence["probe"]
    layout, columns = probe.get("layout"), probe.get("columns")
    if layout in {"two_column", "wide_table"}:
        return str(layout)
    if type(columns) is int and columns >= 2:
        return "wide_table" if columns > 2 else "two_column"
    return "wide_table" if evidence["source_record_count"] > 1 or evidence["max_column_index"] > 1 else "two_column"


def _raw_name(decision: Mapping[str, Any]) -> tuple[str, str]:
    if decision["policy"] == "legacy_id":
        return decision["display_name"], "display_name"
    if decision["override"] is not None:
        return decision["override"], "override"
    if decision["policy"] in {"display", "legacy_id"}:
        return decision["display_name"], "display_name"
    stem = filename_stem(decision["original_filename"])
    if decision["source_layout"] == "two_column":
        return stem, "original_filename"
    probe = decision["layout_evidence"]["probe"]
    label = decision["original_column_label"]
    if probe.get("header_present") is not False and label.strip():
        return label, "original_column_label"
    return f"{stem}__col{decision['source_column_index']:02d}" if stem.strip() else "", "filename_column_index"


def _resolved_base(decision: Mapping[str, Any]) -> tuple[str, list[str]]:
    fallback = f"spectrum__col{decision['source_column_index']:02d}"
    name, notes = _clean(decision["raw_base_name"], fallback)
    if decision["filename_qualifier"]:
        qualifier, qualifier_notes = _clean(filename_stem(decision["original_filename"]), fallback)
        name = qualifier + "__" + name
        notes += ["duplicate wide-table label qualified with filename", *qualifier_notes]
    token = decision["collision_token"]
    budget = BASE_BUDGET - (len(token) + 2 if token else 0)
    clipped = _truncate(name, budget)
    if clipped != name:
        notes.append(f"base truncated on UTF-8 boundary to {budget} bytes (suffix budget reserved)")
    if token:
        clipped += "__" + token
        notes.append("stable identity token resolves name collision")
    return clipped, notes


def allocate_bases(workspace: BatchWorkspace, naming: Naming = None) -> dict[str, dict[str, Any]]:
    options = get_naming_options(workspace, naming)
    result: dict[str, dict[str, Any]] = {}
    by_source: dict[str, list[Any]] = defaultdict(list)
    for record in workspace.records.values():
        by_source[record.source_id].append(record)
    for sid, record in sorted(workspace.records.items()):
        source = workspace.sources[record.source_id]
        evidence = {
            "probe": {key: source.import_probe[key] for key in ("layout", "columns", "header_present") if key in source.import_probe},
            "source_record_count": len(by_source[record.source_id]),
            "max_column_index": max(item.original_column_index for item in by_source[record.source_id]),
        }
        decision: dict[str, Any] = {
            "spectrum_id": sid, "source_id": record.source_id,
            "source_column_index": record.original_column_index,
            "original_filename": source.original_filename,
            "original_column_label": record.original_column_label,
            "display_name": record.display_name, "source_sha256": source.original_bytes_sha256,
            "source_layout": _layout(evidence), "layout_evidence": evidence,
            "policy": options["policy"], "policy_version": NAMING_POLICY_VERSION,
            "override": options["overrides"].get(sid), "collision_token": "",
            "collision_reason": "", "filename_qualifier": False,
        }
        decision["raw_base_name"], decision["name_basis"] = _raw_name(decision)
        decision["base_name"], decision["notes"] = _resolved_base(decision)
        result[sid] = decision

    def groups() -> list[list[dict[str, Any]]]:
        collection: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for decision in result.values():
            collection[_key(decision["base_name"])].append(decision)
        return [items for items in collection.values() if len(items) > 1]

    for group in groups():
        if all(item["name_basis"] == "original_column_label" for item in group) and len({item["source_id"] for item in group}) > 1:
            for decision in group:
                decision["filename_qualifier"] = True
                decision["collision_reason"] = "duplicate column labels across sources"
                decision["base_name"], decision["notes"] = _resolved_base(decision)
    for _ in range(17):
        duplicates = groups()
        if not duplicates:
            return result
        for group in duplicates:
            length = max(len(item["collision_token"]) for item in group)
            length = 8 if length == 0 else length + 4
            if length > 64:
                raise _error("stable full identity token collision; refusing to overwrite")
            for decision in group:
                decision["collision_token"] = _identity_token(decision["spectrum_id"], decision["source_id"])[:length]
                decision["collision_reason"] = "NFC/case-insensitive name collision after cleaning/truncation"
                decision["base_name"], decision["notes"] = _resolved_base(decision)
    raise _error("could not allocate unique export names")


def processing_suffix(branch: str, effective_recipe: Mapping[str, Any] | None = None,
                      parent_effective_recipe: Mapping[str, Any] | None = None) -> str:
    if branch == "coarse":
        return "__baseline-coarse"
    if branch not in {"baseline", "smoothed", "normalized_baseline", "normalized_smoothed"}:
        raise _error("unknown export branch")
    suffix = "__baseline"
    if branch in {"smoothed", "normalized_smoothed"}:
        recipe = effective_recipe if branch == "smoothed" else parent_effective_recipe
        method = (recipe or {}).get("method")
        if not isinstance(method, str) or method not in _SMOOTHING:
            raise _error("unknown committed smoothing method")
        suffix += "__smooth-" + method
    if branch.startswith("normalized_"):
        method = (effective_recipe or {}).get("method")
        if not isinstance(method, str) or method not in _NORMALIZATION:
            raise _error("unknown committed normalization method")
        suffix += "__" + _NORMALIZATION[method]
    return suffix


def _safe_path(path: str, *, bounded: bool = True) -> None:
    parsed = PurePosixPath(path)
    if not path or "\\" in path or parsed.is_absolute() or any(part in {"", ".", ".."} for part in path.split("/")):
        raise _error("unsafe export member path")
    for part in parsed.parts:
        if _clean(part, "spectrum")[0] != part or (bounded and len(part.encode("utf-8")) > MAX_FILENAME_BYTES):
            raise _error("unsafe or overlong export filename")


def make_name_row(decision: Mapping[str, Any], *, branch: str, quantity: str,
                  purpose: str, fingerprint: str, parent_fingerprint: str | None,
                  recipe_json: str, included_as: str,
                  effective_recipe: Mapping[str, Any] | None = None,
                  parent_effective_recipe: Mapping[str, Any] | None = None,
                  components: bool = False, derived_filenames: Sequence[str] = ()) -> dict[str, Any]:
    suffix = processing_suffix(branch, effective_recipe, parent_effective_recipe)
    label = decision["base_name"] + suffix
    row = {key: decision[key] for key in _DECISION_KEYS}
    row.update(branch=branch, quantity=quantity, purpose=purpose, fingerprint=fingerprint,
               parent_fingerprint=parent_fingerprint, included_as=included_as,
               spectra_csv=f"spectra/{label}.csv", recipe_json=recipe_json,
               components_csv=f"components/{label}__components.csv" if components else None,
               removed_csv=f"components/{label}__removed_component.csv" if branch == "smoothed" else None,
               derived_outputs={name: f"spectra/{label}__{name}" for name in derived_filenames},
               wide_column_label=label)
    if included_as not in {"requested", "parent"}:
        raise _error("invalid inclusion role")
    for path in _paths(row):
        _safe_path(path)
    return row


def _paths(row: Mapping[str, Any]) -> list[str]:
    return [row[key] for key in ("spectra_csv", "recipe_json", "components_csv", "removed_csv") if row.get(key) is not None] + list(row.get("derived_outputs", {}).values())


def validate_name_row(row: Mapping[str, Any], source: Mapping[str, Any], **node: Any) -> None:
    decision = {key: row[key] for key in _DECISION_KEYS}
    for key in ("spectrum_id", "source_id", "source_column_index", "original_filename", "display_name", "source_sha256"):
        if decision[key] != source[key]:
            raise _error(f"name map/source mismatch: {key}")
    if decision["original_column_label"] != source["source_column_label"]:
        raise _error("name map/source column mismatch")
    if decision["policy"] not in POLICIES[:2] or decision["policy_version"] != NAMING_POLICY_VERSION:
        raise _error("unsupported readable naming policy")
    validate_naming_preferences({"export_naming": {"policy": decision["policy"], "override": decision["override"]}})
    evidence = decision["layout_evidence"]
    if not isinstance(evidence, dict) or set(evidence) != {"probe", "source_record_count", "max_column_index"}:
        raise _error("invalid layout evidence")
    if not isinstance(evidence["probe"], dict) or not evidence["probe"].keys() <= {"layout", "columns", "header_present"}:
        raise _error("invalid import probe evidence")
    probe = evidence["probe"]
    if ("layout" in probe and probe["layout"] not in {"two_column", "wide_table"}) or ("columns" in probe and (type(probe["columns"]) is not int or probe["columns"] < 2)) or ("header_present" in probe and type(probe["header_present"]) is not bool):
        raise _error("invalid import probe naming fields")
    if any(type(evidence[key]) is not int or evidence[key] < 1 for key in ("source_record_count", "max_column_index")):
        raise _error("invalid fallback layout counts")
    columns = probe.get("columns")
    if columns is not None and (
        evidence["max_column_index"] >= columns or evidence["source_record_count"] >= columns
        or ("layout" in probe and probe["layout"] != ("two_column" if columns == 2 else "wide_table"))
    ):
        raise _error("contradictory layout/column evidence")
    if probe.get("layout") == "two_column" and (evidence["max_column_index"] > 1 or evidence["source_record_count"] > 1):
        raise _error("two-column source contains multiple intensity columns")
    if evidence["max_column_index"] < decision["source_column_index"] or _layout(evidence) != decision["source_layout"]:
        raise _error("layout decision mismatch")
    raw, basis = _raw_name(decision)
    if (raw, basis) != (decision["raw_base_name"], decision["name_basis"]):
        raise _error("name origin mismatch")
    token = decision["collision_token"]
    if not isinstance(token, str) or (token and (len(token) not in range(8, 65, 4) or token != _identity_token(decision["spectrum_id"], decision["source_id"])[:len(token)])):
        raise _error("invalid stable collision token")
    qualifier = decision["filename_qualifier"]
    if type(qualifier) is not bool or (qualifier and basis != "original_column_label"):
        raise _error("invalid filename qualification")
    expected_reason = ("NFC/case-insensitive name collision after cleaning/truncation" if token else "duplicate column labels across sources" if qualifier else "")
    if decision["collision_reason"] != expected_reason:
        raise _error("collision reason mismatch")
    base, notes = _resolved_base(decision)
    if (base, notes) != (decision["base_name"], decision["notes"]):
        raise _error("resolved name mismatch")
    if dict(row) != make_name_row(decision, **node):
        raise _error("name map path/branch/identity mismatch")


def ensure_unique_names(rows: Sequence[Mapping[str, Any]]) -> None:
    paths: set[str] = set()
    identities: set[tuple[str, str]] = set()
    labels: set[str] = set()
    decisions: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    source_spectra: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        identity = row["spectrum_id"], row["branch"]
        decision = {key: row[key] for key in _DECISION_KEYS}
        if identity[0] in decisions and decisions[identity[0]] != decision:
            raise _error("branches of one spectrum have inconsistent naming decisions")
        decisions[identity[0]] = decision
        source = {key: row[key] for key in ("original_filename", "source_sha256", "source_layout", "layout_evidence")}
        source_id = row["source_id"]
        if source_id in sources and sources[source_id] != source:
            raise _error("columns of one source have inconsistent naming evidence")
        sources[source_id] = source
        source_spectra[source_id].add(identity[0])
        if len(source_spectra[source_id]) > source["layout_evidence"]["source_record_count"]:
            raise _error("source record count contradicts exported identities")
        if identity in identities or _key(row["wide_column_label"]) in labels:
            raise _error("duplicate export node or wide column label")
        identities.add(identity)
        labels.add(_key(row["wide_column_label"]))
        for path in _paths(row):
            _safe_path(path)
            key = _key(path)
            if key in paths:
                raise _error("duplicate export member; refusing overwrite")
            paths.add(key)


def name_map_table(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    headers = sorted({key for row in rows for key in row})
    values = [[json.dumps(row.get(key), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
               if isinstance(row.get(key), (Mapping, list, tuple)) else row.get(key)
               for key in headers] for row in rows]
    return headers, values


def naming_plan_digest(bases: Mapping[str, Any], rows: Sequence[Mapping[str, Any]] = ()) -> str:
    payload = {"bases": bases, "nodes": sorted(rows, key=lambda row: (row["spectrum_id"], row["branch"]))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
