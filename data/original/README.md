# Local raw data directory

This public repository intentionally does not distribute experimental spectra.
Place your own supported `.dpt`, `.csv`, or `.txt` files in this directory for
local analysis, or upload them through the Streamlit interface.

Everything in this directory except this README is ignored by Git. Do not force
add confidential spectra to a commit.

For a directory of two-column DPT files, filenames may contain numeric
perturbation values (for example, `0MIN.dpt`, `5MIN.dpt`, and `10MIN.dpt`). The
importer can sort those values numerically. A file named `BASELINE.dpt` is
excluded from the named demo workflow unless you explicitly load it elsewhere.
