# swrc-ptf: soil water retention model fitting, AICc model support, and support-stratified pedotransfer functions

This repository accompanies a paper currently under submission.

- Seki, Katsutoshi. Measurement-conditioned support for retention-model complexity and its
implications for soil hydraulic pedotransfer functions. Available at SSRN:
http://dx.doi.org/10.2139/ssrn.7411951

It is designed to ensure
full transparency and reproducibility by allowing readers to reproduce the entire
analysis — every model fit, figure, and numeric table reported in the paper — from the
openly available input data, by running a single script (`run.py`). Nothing in the paper's
computation is hidden: the code that produced each result is included here as a documented
function, and the same script regenerates the results, figures, and tables end to end.

Running `run.py` reproduces the full computation of the paper — fitting the van
Genuchten (VG) and dual-VG-CH (DVC) retention models to every curve of the GSHP
database, deciding which model each curve's measurements support (AICc), and building
and cross-validating the support-stratified pedotransfer functions — and regenerates the
figures and numeric tables.

## What run.py produces

`python3 run.py` reads the GSHP retention data in `data/` and writes:

- `result/` — computed results: the fitted parameters, the three-group model-support
  tables, the PTF leave-one-reference-out (LORO) errors, the study-clustered bootstrap
  confidence intervals, and the supporting analyses (degeneracy, identifiability,
  downsampling, w1 profiles).
- `fig/` — the computed manuscript figures (`fig2.svg` … `fig7.svg`).
- `table/` — the numeric tables computed from the data: `table1` (three-group counts),
  `table3` (reconstructed-θ(h) LORO micro RMSE), `tableS1` (curve accounting),
  `tableS3` (per-study LORO folds for DVC-supported evaluation), `tableS4` (apparent RMSE).

All analysis code is in a single self-contained script, `run.py`; each stage is a
function with a docstring, and `main()` calls them in order.

## Usage

1. Install the dependencies (versions unpinned; current releases):

   ```
   python3 -m pip install -r requirements.txt
   ```

2. Clone this repository:

   ```
   git clone https://github.com/sekika/swrc-ptf
   cd swrc-ptf
   ```

3. Obtain the input data and place it in `data/`.

   The Global Soil Hydraulic Properties (GSHP) database is openly available on Zenodo
   (record 6640246, concept DOI [10.5281/zenodo.5547338](https://doi.org/10.5281/zenodo.5547338),
   CC BY 4.0; Gupta et al., 2022). Download it and put the water-retention CSV at:

   ```
   data/WRC_dataset_surya_et_al_2021_final.csv
   ```

   The data are not redistributed here; users obtain them from Zenodo.

4. Run the full pipeline:

   ```
   python3 run.py
   ```

   The fitting stages fit ~13,800 curves with two models each, so a full run takes a
   while. Results are deterministic up to BLAS/LAPACK last-bit differences between
   environments (this affects only unrounded floats; all reported/rounded values are
   stable).

## Model definitions (unsatfit)

- VG: `set_model('VG', const=['q=1'])` — free `θ_r, θ_s, α, n` (k = 4).
- DVC: `set_model('dual-VG-CH', const=['qr=0','q=1'])` — free `θ_s, α, w₁, n₁, n₂`,
  with `θ_r = 0` and a common `α` (k = 5). The two modes are ordered so that `n₁ > n₂`.

Model support is decided by the corrected AIC (`aicc_ht`), which is undefined
(`None`) when `N − k − 1 ≤ 0`; such curves are labelled AICc-uncomparable.

## Data reference

Gupta, S., Papritz, A., Lehmann, P., Hengl, T., Bonetti, S., Or, D., 2022. Global Soil
Hydraulic Properties dataset based on legacy site observations and robust
parameterization. Scientific Data 9, 381. https://doi.org/10.1038/s41597-022-01481-5
