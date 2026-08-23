# Private-data handling

Experimental spectra are intentionally excluded from the public repository and
from every published Git commit. The local `data/original/` directory is ignored
except for its README, so private inputs remain available on the workstation
without being uploaded to GitHub.

## Supported local inputs

- A wide CSV/TXT table with a wavenumber column and one or more spectrum columns.
- Multiple headerless, two-column DPT/CSV/TXT spectra with matching axes.
- Explicitly confirmed intensity units: absorbance, percent transmittance, or
  fractional transmittance.

When numeric perturbation values are present in filenames, the directory loader
can sort them numerically. A file named `BASELINE.dpt` is excluded by the demo
workflow and is never silently interpreted as an algorithmic baseline.

## Publication rule

Do not use `git add -f` for anything under `data/original/`. Private replay ZIPs,
`.ftirw` projects, generated figures, and data-derived release manifests should
remain in ignored local output locations. Synthetic examples under `examples/`
are the only datasets distributed with the repository.
