# Unified architecture

## Active scientific path

```text
SpectrumSet (explicit unit and perturbation order)
  │
  └─ ftir_baseline.run_pipeline
       ├─ raw / absorbance / selected range
       ├─ estimate-only smoothing channel
       ├─ coarse + fine baseline
       ├─ corrected absorbance = unsmoothed absorbance - total baseline
       ├─ QC and optional display/sensitivity branches
       └─ PipelineResult.analysis_data
             │
             └─ PreparedSpectralDataset (immutable checkpoint)
                  ├─ baseline-only bundle and stop
                  ├─ CSV + metadata reload checkpoint
                  └─ TwoDCOSWorkflowService
                       ├─ range masks only
                       ├─ ftir2dcos.twodcos.compute_2dcos
                       └─ ftir2dcos.twodcos.compute_cross_2dcos
```

The coordination package does not implement unit conversion, smoothing, baseline estimation, normalization, a Noda matrix, or correlation formulas.

## v0.2 view and state layer

```text
Committed PipelineConfig
  ├─ Coarse/Fine draft controls
  │    ├─ Preview -> frozen run_pipeline over the complete series
  │    │              -> isolated preview result
  │    └─ Adopt/Apply -> committed config -> v0.1 descendant invalidation
  │
  └─ formal baseline run -> PipelineResult / QCResult
       ├─ Series Consistency & QC (read-only views)
       ├─ display_units (A, T, %T copies only)
       ├─ independent derived T/%T CSV downloads
       └─ PreparedSpectralDataset -> 2D-COS
                                    └─ cross_views
                                         ├─ stored orientation
                                         ├─ deterministic reverse orientation
                                         └─ full N×N block overview
```

Preview uses the authoritative pipeline because collaborative and shared-shape modes depend on the complete series. Selecting a representative spectrum only chooses which already-produced row or UI aggregate is plotted. Preview results are not baseline export inputs.

The Series Consistency & QC page consumes an existing `PipelineResult` and `QCResult`. Its five heatmaps, complete table, trends and drill-down neither invoke a second QC implementation nor mutate, delete, reorder or clip result arrays.

`ftir_workbench.display_units` owns the view-only inverse representations `T=10^-A` and `%T=100×10^-A`. It returns owned finite float64 arrays, rejects complex/NaN/Inf/overflow, permits `%T>100` for negative A and never clips. These copies do not enter Prepared data, fingerprints or the frozen baseline ZIP.

## Cross orientation contract

`TwoDCOSWorkflowService` continues to compute cross results from `combinations(config.ranges, 2)`. For n ranges this creates `C(n,2)` `CrossRangeResult` objects. `ftir_workbench.cross_views` exposes two orientations per pair, for `n(n-1)` oriented maps in total, without calling `compute_cross_2dcos` again:

```text
Phi_reverse = Phi_stored.T
Psi_reverse = -Psi_stored.T
```

The stored row range is not assumed to be `first_range`. `row_variable` and `column_variable` map `nu1` to `first_range` and `nu2` to `second_range`; reverse swaps both variables and axes. The full block overview uses self matrices on the diagonal, stored matrices in their actual off-diagonal cells and reverse matrices in the opposite cells. Cross orientation is a matrix layout, not a causal arrow.

## Legacy compatibility

`ftir2dcos.pipeline`, `ftir2dcos.preprocessing`, and `ftir2dcos.conversion` remain available so existing scripts and their regression tests continue to run. They are not imported by the unified UI or workbench services. A source audit test enforces this boundary.

## Fingerprints

Fingerprints use SHA-256 over versioned canonical JSON and named little-endian float64 arrays. Shapes and field names are part of the digest, so reshaping or exchanging axes cannot preserve the same fingerprint accidentally.

- The baseline fingerprint identifies the baseline scientific result and recipe.
- The prepared-data digest identifies the exact wavenumber, perturbation, labels, spectra and normalization state supplied downstream.
- The 2D scientific fingerprint includes prepared parent identity, ranges, convention, grid strategy and nonuniform perturbation policy.
- Display settings are excluded from scientific fingerprints.

## Axis and ordering policy

- A wavenumber axis must be strictly monotonic and may remain ascending or descending.
- Spectra are always rows and wavenumbers are always columns.
- Perturbation labels, numeric values and rows remain aligned.
- Directory input has no portable acquisition order; numeric sorting is explicit and recorded.
- Nonuniform perturbation coordinates are warned or rejected according to configuration. The first release uses acquisition index order for the Hilbert–Noda matrix and does not claim nonuniform-time weighting.

## Invalidation

| Change | Baseline | Prepared | 2D |
|---|---:|---:|---:|
| Raw data, unit or perturbation order | invalidate | invalidate | invalidate |
| Baseline range/method/anchors/scientific normalization | invalidate | invalidate | invalidate |
| 2D ranges, convention or grid policy | keep | keep | invalidate |
| Contours, percentile, font, line width or display normalization | keep | keep | keep |
| Coarse/Fine draft or Preview representative | keep | keep | keep |
| A/%T/T display selection | keep | keep | keep |
| Cross pair/orientation focus or numeric preview size | keep | keep | keep |

## Export lineage

Every 2D manifest records the parent baseline run id, baseline fingerprint and prepared-data digest. Baseline and 2D ZIP files contain a per-member SHA-256 manifest. A `.ftirw` project archive is a ZIP container over those independently verifiable run bundles and a project configuration snapshot.

The v0.2 2D bundle keeps the v0.1 stored cross filenames and adds reverse synchronous/asynchronous CSV files plus `orientations.json`. The verifier checks transpose identities, swapped axes, variable/range metadata and pair/orientation counts. A complete v0.1 bundle without these additive members remains valid. The baseline bundle itself is unchanged; derived T/%T files are independent downloads.

## Frozen and excluded architecture

The v0.2 baseline freeze covers `src/ftir_baseline/**`, `tests/baseline_regression/**` and `legacy/baseline_streamlit_app.py`. This release does not introduce multiple Baseline Blocks, local range correction, processing/analysis dual ranges, `AnalysisRangePreparationService`, global/local-fine branches, sensitivity Run A/B/C, new baseline methods/config fields/workflow states, endpoint forcing, a desktop shell or a large UI rewrite.
