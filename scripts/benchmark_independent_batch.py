#!/usr/bin/env python3
"""Measure synthetic independent imports, arPLS previews, and workspace caching.

Default acceptance workload: 100 spectra x 4000 coordinates. All data is created
in memory from a fixed RNG seed; the script never reads experimental files or
writes spectrum arrays. The optional output is a measurement-only JSON report.
Run this benchmark during Phase 5 after the functional acceptance phases.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ftir_baseline.config import CoarseBaselineConfig, PipelineConfig
from ftir_baseline.models import SpectrumSet
from ftir_baseline.pipeline import PipelineResult, run_pipeline
from ftir_workbench.batch.fingerprints import implementation_fingerprint
from ftir_workbench.batch.importing import import_sources
from ftir_workbench.batch.models import BatchWorkspace, StageSnapshot
from ftir_workbench.batch.service import preview_coarse
from ftir_workbench.batch.state import confirm_coarse, set_coarse_draft


def _peak_rss_mib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # getrusage reports bytes on macOS and KiB on Linux; preserve the unit evidence.
    divisor = 1024.0**2 if platform.system() == "Darwin" else 1024.0
    return raw / divisor


def _synthetic_inputs(count: int, points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(4000.0, 400.0, points)
    rows = []
    for _ in range(count):
        scaled = (x - x.min()) / np.ptp(x)
        baseline = rng.uniform(0.03, 0.2) + rng.uniform(0.02, 0.15) * scaled
        baseline += rng.uniform(0.01, 0.06) * scaled**2
        y = baseline.copy()
        for center, width in ((3400.0, 115.0), (1730.0, 24.0), (1150.0, 48.0)):
            y += rng.uniform(0.03, 0.4) * np.exp(-((x - center) / width) ** 2)
        y += rng.normal(0.0, 0.001, points)
        rows.append(y)
    return x, np.array(rows, dtype=np.float64)


def _csv_bytes(x: np.ndarray, spectra: np.ndarray) -> bytes:
    stream = io.StringIO()
    np.savetxt(
        stream, np.column_stack((x, spectra.T)), fmt="%.17g", delimiter=",",
        header="wavenumber," + ",".join(f"synthetic_{index:03d}" for index in range(len(spectra))),
        comments="",
    )
    return stream.getvalue().encode("utf-8")


def _numeric_arrays(snapshot: StageSnapshot) -> dict[str, np.ndarray]:
    result = snapshot.result
    return {
        "wavenumber": result.absorbance_selected.wavenumber,
        "raw_absorbance": result.absorbance_selected.spectra,
        "estimation_channel": result.baseline_estimation_spectra,
        "coarse_baseline": result.baseline.coarse_baseline,
        "fine_baseline": result.baseline.fine_baseline,
        "total_baseline": result.baseline.total_baseline,
        "corrected_absorbance": result.analysis_data,
    }


def benchmark(*, count: int, points: int, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    rss_at_start = _peak_rss_mib()
    generation_started = time.perf_counter()
    x, y = _synthetic_inputs(count, points, seed)
    contents = _csv_bytes(x, y)
    generation_seconds = time.perf_counter() - generation_started
    workspace = BatchWorkspace()
    import_started = time.perf_counter()
    ids = import_sources(workspace, [("synthetic_100x4000.csv", contents)])
    import_seconds = time.perf_counter() - import_started
    if len(ids) != count or workspace.import_issues:
        raise AssertionError(f"synthetic import failed: {workspace.import_issues}")
    if any(workspace.records[sid].wavenumber.size != points for sid in ids):
        raise AssertionError("import changed the independent coordinate count")
    calls = 0

    def counted_runner(data: SpectrumSet, config: PipelineConfig) -> PipelineResult:
        nonlocal calls
        calls += 1
        if data.n_spectra != 1:
            raise AssertionError("benchmark encountered a non-singleton scientific call")
        if config.series_mode != "independent_locked" or config.normalization.method != "none":
            raise AssertionError("benchmark encountered a forbidden independent recipe")
        return run_pipeline(data, config)

    per_spectrum_seconds = []
    processing_started = time.perf_counter()
    for index, sid in enumerate(ids):
        if workspace.states[sid].coarse_draft.method != "arpls":
            raise AssertionError("benchmark requires the current arPLS starting recipe")
        item_started = time.perf_counter()
        preview_coarse(workspace, sid, runner=counted_runner)
        confirm_coarse(workspace, sid)
        per_spectrum_seconds.append(time.perf_counter() - item_started)
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"Synthetic coarse previews confirmed: {index + 1}/{count}", file=sys.stderr)
    processing_seconds = time.perf_counter() - processing_started
    calls_after_processing = calls
    repeated_started = time.perf_counter()
    same_snapshot_objects = True
    for sid in ids:
        original = workspace.states[sid].coarse_snapshot
        repeated = preview_coarse(workspace, sid, runner=counted_runner)
        same_snapshot_objects = same_snapshot_objects and repeated is original
    repeated_seconds = time.perf_counter() - repeated_started
    cache_additional_calls = calls - calls_after_processing
    if calls_after_processing != count or cache_additional_calls != 0 or not same_snapshot_objects:
        raise AssertionError("repeat preview cache did not preserve each confirmed result")

    first_id = ids[0]
    first = workspace.states[first_id].coarse_snapshot
    if first is None:
        raise AssertionError("first spectrum has no confirmed coarse result")
    alone = BatchWorkspace()
    alone_id = import_sources(alone, [("synthetic_A_alone.csv", _csv_bytes(x, y[:1]))])[0]
    alone_started = time.perf_counter()
    alone_snapshot = preview_coarse(alone, alone_id, runner=counted_runner)
    confirm_coarse(alone, alone_id)
    alone_seconds = time.perf_counter() - alone_started
    first_arrays = _numeric_arrays(first)
    alone_arrays = _numeric_arrays(alone_snapshot)
    comparisons = {
        name: {
            "array_equal": bool(np.array_equal(values, alone_arrays[name])),
            "maximum_absolute_difference": float(np.max(np.abs(values - alone_arrays[name]))),
        }
        for name, values in first_arrays.items()
    }
    if not all(item["array_equal"] for item in comparisons.values()):
        raise AssertionError(f"A alone and A in unrelated batch differ: {comparisons}")
    before_mutation_calls = calls
    if len(ids) > 1:
        set_coarse_draft(workspace, ids[1], CoarseBaselineConfig(method="linear"))
        del workspace.records[ids[1]]
        del workspace.states[ids[1]]
        workspace.display_order.remove(ids[1])
    workspace.display_order.reverse()
    unchanged = preview_coarse(workspace, first_id, runner=counted_runner)
    mutation_unchanged = unchanged is first and unchanged.fingerprint == first.fingerprint
    if not mutation_unchanged or calls != before_mutation_calls:
        raise AssertionError("editing/deleting B or reordering the list invalidated A")
    # Public reports contain measurements and synthetic identifiers only, never private paths.
    return {
        "status": "pass",
        "data_scope": "Fixed-seed synthetic spectra only; no experimental input read or saved.",
        "workload": {"spectra": count, "points_per_spectrum": points, "seed": seed,
                     "input_layout": "one independent wide CSV", "input_unit": "absorbance",
                     "input_bytes": len(contents), "coarse_method": "arpls"},
        "environment": {"python": platform.python_version(), "system": platform.system(),
                        "machine": platform.machine(), "implementation_fingerprint": implementation_fingerprint()},
        "seconds": {
            "generate_and_serialize_synthetic_input": generation_seconds,
            "import_probe_parse_and_split": import_seconds,
            "all_coarse_previews_and_confirmations": processing_seconds,
            "per_spectrum_median": float(np.median(per_spectrum_seconds)),
            "per_spectrum_maximum": max(per_spectrum_seconds),
            "repeat_all_previews_from_cache": repeated_seconds,
            "A_alone_preview_and_confirmation": alone_seconds,
            "total": time.perf_counter() - started,
        },
        "process_peak_rss_mib": {"at_start": rss_at_start, "at_finish": _peak_rss_mib(),
                                 "scope": "Whole benchmark process peak, including imports, parser and NumPy arrays; not incremental workspace memory."},
        "pipeline_calls": {"batch_coarse": calls_after_processing,
                           "repeated_previews_additional": cache_additional_calls,
                           "A_alone": 1,
                           "after_B_edit_delete_and_reorder_additional": calls - before_mutation_calls,
                           "total": calls},
        "cache_reused_exact_snapshot_objects": same_snapshot_objects,
        "A_independence": comparisons,
        "A_stage_fingerprint_unchanged_after_B_edit_delete_and_reorder": mutation_unchanged,
        "provenance": "A-alone and batch source identities differ intentionally; numerical arrays are compared exactly.",
        "cache_scope": "One coarse preview and one confirmed coarse reference per record; workspace-local only.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra", type=int, default=100)
    parser.add_argument("--points", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=3030)
    parser.add_argument("--output", type=Path, help="Optional measurement-only JSON output path")
    args = parser.parse_args()
    if args.spectra < 2 or args.points < 10:
        parser.error("use at least 2 spectra and 10 points")
    report = benchmark(count=args.spectra, points=args.points, seed=args.seed)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
