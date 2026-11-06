# Predicting Conduction Velocity from 7T Microstructure MRI

Code for:

> Asadi S., et al. (2026). **Non-invasive prediction of conduction velocities in the human brain from MRI-derived microstructure features at 7 Tesla.** *Imaging Neuroscience*.

If you use this code, please cite this paper.

## Overview

This repository contains the code used to:

1. **Preprocess 7T diffusion MRI** data: multi-shell and multi-echo-time (multi-TE) acquisitions.
2. **Relate white matter microstructure to cortico-cortical conduction properties**: correlations and regression models between microstructure (SMI metrics, axon radius, qT1/R1) and conduction velocity/delay derived from the F-TRACT atlas.
3. **Predict conduction velocity and delay** for all cortico-cortical connections, and reproduce the corresponding figures.


## Pipeline

```
raw dMRI
  │  preprocessing/
  v
microstructure maps (SMI, axon radius, qT1)
  │  see Methods
  v
connectivity matrices
  │  conduction-prediction/01_microstructure-velocity_stats.R
  v
edge tables + statistics
  │  conduction-prediction/02_velocity-prediction_figures.py
  v
predicted velocity/delay + figures

```
The steps between preprocessing and the statistics (microstructure model fitting, tractography and connectivity matrices, F-TRACT conduction delays) use published tools and are described in the Methods.

All connectivity matrices (SMI metrics, axon radius, qT1, R1, fibre length) are defined on the white matter bundle atlas at scale 1 of the multiconn atlas (Alemán-Gómez et al., 2022); the analyses use its 68 cortical regions.

### 1. Preprocessing (`preprocessing/`)

Processing steps are described in the Methods.

- `dwi_preproc_ms.py`: multi-shell data (NORDIC denoising, Gibbs ringing correction, `topup`, `eddy`, DTI fit on b ≤ 1000 s/mm²).
- `dwi_preproc_mt.py`: multi-TE data (TE = 80, 100, 120, 135 ms); the same steps for each TE, plus additional motion correction and the spherical-mean 4D input for axon radius estimation.
- `preproc_fn.py`: functions used by both scripts. `sigma_map.m`: noise map after NORDIC denoising.

NORDIC ([`NIFTI_NORDIC.m`](https://github.com/SteenMoeller/NORDIC_Raw), v1.1; Moeller et al., 2021) is not included in this repository; set its location in `preproc_fn.py`.

### 2. Statistics (`conduction-prediction/`)

1. `01_microstructure-velocity_stats.R`
   - **Input:** group-average connectivity matrices (CSV) and region order (Zenodo), plus the F-TRACT delay matrix (`conn_delays_dcm_masked_by_counts_thr_10.csv`, not included; obtain from F-TRACT) in the same folder.
   - **Output:** edge tables `filtered_micro_delay_dcm10thr_ctx.csv` and `all_micro_delay_dcm10thr_ctx.csv`.
2. `02_velocity-prediction_figures.py`
   - **Input:** the two edge tables.
   - **Output:** predicted velocity and delay matrices (CSV) and the figure panels.

## Requirements

**Software**

| Tool | Used for | Version |
|---|---|---|
| FSL (incl. `eddy_cuda10.2`, GPU) | topup, eddy, dtifit, fslroi/fslmerge/fslmaths | 6.0.7.11 |
| MRtrix3 | `dwiextract`, `mrdegibbs` | 3.0.4 |
| FreeSurfer | `mri_synthstrip` | 7.4.1 |
| MATLAB | NORDIC denoising | R2021b |
| R | statistics | 4.6.1 |
| Python | preprocessing and prediction | 3.10 |
| ANTsPy (antspyx, Python) | motion correction of the multi-TE data | 0.6.2 |

**Python packages:** numpy, pandas, nibabel, antspyx, scipy, statsmodels, scikit-learn, matplotlib, seaborn

**R packages:** ggplot2, ggpubr, GGally, psych, corrplot, ggfortify, car, effectsize, lm.beta, dplyr

## Data

Available on Zenodo (https://doi.org/10.5281/zenodo.23954112): connectivity matrices (83 × 83, scale-1 multiconn atlas; only the 68 cortical regions are analysed), region order, predicted conduction velocity and delay matrices, and group-average microstructure maps (NIfTI, MNI space).

Axonal conduction delay data come from the F-TRACT atlas (Lemaréchal et al., 2022; Trebaul et al., 2018; https://f-tract.eu).

## Contact
Saina Asadi, saina.asadi@outlook.com
