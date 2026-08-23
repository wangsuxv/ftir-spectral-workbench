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

## Export lineage

Every 2D manifest records the parent baseline run id, baseline fingerprint and prepared-data digest. Baseline and 2D ZIP files contain a per-member SHA-256 manifest. A `.ftirw` project archive is a ZIP container over those independently verifiable run bundles and a project configuration snapshot.
