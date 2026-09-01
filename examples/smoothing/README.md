# Synthetic post-baseline smoothing example

This directory contains a tiny, deterministic `PreparedSpectralDataset` checkpoint:

- `synthetic_corrected_prepared.csv` — 3 synthetic corrected-absorbance spectra × 13 uniformly spaced wavenumbers;
- `prepared_spectrum.meta.json` — the required provenance and SHA-256 sidecar.

The values are hand-authored synthetic curves for documentation and automated tests; they are not sampled, aggregated, or derived from an experiment. They contain no experimental spectra, original instrument data, sample/person identifiers, acquisition metadata, instrument serial numbers, private paths, or private project fingerprints.

Load it with the public Prepared loader:

```python
from ftir_workbench.export import load_prepared

prepared = load_prepared(
    "examples/smoothing/synthetic_corrected_prepared.csv"
)
assert prepared.spectra.shape == (3, 13)
```

Create and verify a smoothing bundle:

```bash
ftir-workbench smooth \
  examples/smoothing/synthetic_corrected_prepared.csv \
  --method savgol --window-length 7 --polyorder 2 --mode interp \
  --output outputs/smoothing-example

ftir-workbench verify \
  outputs/smoothing-example/post_baseline_smoothing_run.zip
```

The example starts at corrected absorbance. It is not a raw FTIR import fixture and is not intended for evaluating a scientifically optimal smoothing method.

A bundle created from real input still contains the complete baseline-corrected spectra and derived smoothing output even though it contains no raw instrument file. Keep such bundles in ignored output directories or private storage; only the synthetic files in this directory are intended to be safe repository examples.
