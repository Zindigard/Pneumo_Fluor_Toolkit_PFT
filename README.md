# Pneumo-Fluor-Toolkit

Pneumo-Fluor Toolkit is a deep learning–based fluorescence image analysis pipeline for quantitative single-cell analysis of Streptococcus pneumoniae. It integrates denoising, segmentation, feature extraction, and visualization for 2D/3D and time-lapse microscopy data, with a focus on reproducible and scalable analysis via a napari plugin.

## Data and aims
The toolkit is designed to support microscopy datasets such as:
- **2D WGA + DAPI** images (cell wall + nucleoid organization)
- **3D SIM stacks** of peptidoglycan labeling (e.g., NADA/RADA/HADA conditions)
- **2D time-lapse HADA** imaging for cell wall dynamics and growth

Core goals:
- Provide robust preprocessing (background suppression, normalization, denoising)
- Perform single-cell segmentation and mask extraction
- Extract quantitative single-cell features (morphology, intensity distributions, spatial patterns)
- Enable fast visual inspection and batch processing inside napari(will be done)

## Repository structure
- `src/PFT/` — installable Python package (core logic + napari integration)
- `tests/` — tests if file gives correct version of output
- `models/`,`results/` — local model and outputs (ignored by git, will be added as final versions of model(maybe))
- `scripts` - small scripts that allow visualization of data and manz other options(in future just for python checking)


## Installation (dev)
Create/activate your environment, then install in editable mode:

```bash
pip install -e .