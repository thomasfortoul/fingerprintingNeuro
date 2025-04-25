# MEG Spectral Fingerprinting in Alzheimer’s Disease  
*Codebase for “MEG Spectral Fingerprinting in Alzheimer’s Disease: A Longitudinal and Cross-Sectional Analysis” (Fortoul 2025)*  

---

## 1 . Overview  
This repository contains the complete analysis pipeline for the paper.  
Workflow outline:  

1. **Fingerprint extraction** — convert resting‑state MEG power‑spectral density (PSD) into a subject‑specific spectral fingerprint.  
2. **Fingerprint metrics** — compute self‑similarity and *differentiability* (**D**) for every participant.  
3. **Identification modelling** — fit a single‑predictor logistic model that maps **D** to identification probability; benchmark accuracy with permutation tests.  
4. **Longitudinal modelling** — relate fingerprint drift to months‑to‑diagnosis (T<sub>dx</sub>).  

All figures and statistics in the manuscript are reproduced by these scripts.

---

## 2 . Repository structure

| File | Purpose |
|------|---------|
| **fingerprinting.R** | Converts raw MEG PSD files into per‑subject spectral fingerprints (CSV). |
| **similarity_differentiability.py** | Computes self‑similarity, **D**, and similarity matrices; saves summary tables & plots. |
| **differentiation_accuracy.py** | Calculates single‑match identification accuracy, fits the logistic model, draws ROC & decision‑boundary plots. |
| **permutation_Analysis.py** | Performs 1 000 random‑subsampling permutations to test cohort‑size effects. |
| **longitudinal_Analysis.py** | Main longitudinal regressions of fingerprint metrics vs. T<sub>dx</sub>. |
| **longitudinal_Analysis2.py** | Robust / mixed‑effects variants and sensitivity checks (supplementary). |

---

## 3 . Requirements

| Language | Version | Key packages |
|----------|---------|--------------|
| **Python** | ≥ 3.10 | numpy, pandas, scipy, scikit‑learn, statsmodels, matplotlib, seaborn |
| **R** | ≥ 4.2 | tidyverse, data.table, eegUtils (or MNE‑R equivalent) |

Create an isolated environment (example with conda):  

```bash
conda env create -f environment.yml        # or install manually
pip install -r requirements.txt            # Python deps
```

---

## 4 . Quick start

```bash
# 1. Clone the repo and move inside
git clone https://github.com/<user>/AD-MEG-fingerprinting.git
cd AD-MEG-fingerprinting

# 2. Place raw PSD/metadata in data/  (see data/README.md for layout)

# 3. Build fingerprints
Rscript fingerprinting.R

# 4. Run the core analyses  (order matters)
python similarity_differentiability.py
python differentiation_accuracy.py
python permutation_Analysis.py
python longitudinal_Analysis.py

# 5. (Optional) Supplementary checks
python longitudinal_Analysis2.py
```

Outputs (plots & CSV summaries) are written to `results/` and can be dropped straight into the LaTeX manuscript.

---

## 5 . Reproducing paper numbers  
Default parameters (seed = 42, 1 000 permutations, 10 % FDR) match the submission settings.  
Paths and hyper‑parameters can be overridden via the argument parser at the top of each script.

---

## 6 . Contact  
**Thomas Fortoul** — thomas.fortoul@mail.mcgill.ca  
NeuroSpeed Lab, Montreal Neurological Institute (MNI), McGill University
