# Reproducible private artifacts

Real-data ZIP/`.ftirw` files are generated into the ignored `outputs/` directory.
Data-derived release manifests are also ignored so that public Git history does
not reveal fingerprints or measurements from private spectra.

With your own local data, regenerate and validate from the repository root:

```bash
ftir-workbench demo --input-dir data/original --output outputs/real-data-demo
python scripts/validate_real_data_demo.py outputs/real-data-demo
```
