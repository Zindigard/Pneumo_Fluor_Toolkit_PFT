# Pneumo-Fluor Toolkit

Pneumo-Fluor Toolkit (PFT) is a reproducible fluorescence-microscopy analysis
pipeline for quantitative single-cell investigation of *Streptococcus
pneumoniae*. The project combines raw-image import, metadata-preserving
OME-Zarr storage, denoising and restoration, instance segmentation,
cell-standardization, feature extraction, statistical testing, and
publication-oriented visualization.

The toolkit supports command-line batch processing and is structured for later
integration into a napari interface.

## Scientific scope

PFT is designed for the following  dataset classes:

- **2D time-series HADA imaging** for analysis of cell-wall labeling and growth.
- **2D WGA-DAPI imaging** for combined cell-wall and nucleoid analysis.
- **3D structured-illumination microscopy stacks** containing fluorescent
  peptidoglycan labels such as HADA, NADA, RADA, and TADA.

The pipeline is intended to preserve the relationship between raw microscopy
data, processing parameters, trained models, predicted masks, quantitative
measurements, statistical results, and final figures.

## Main capabilities

1. **CZI import and metadata preservation**
   - Recursive raw-data discovery.
   - CZI pixel and metadata extraction.
   - Exact non-normalized level-0 OME-Zarr preservation.
   - Physical sampling, channel, microscope, objective, detector, and SIM
     metadata storage.
   - Pixel-level and metadata-level validation reports.

2. **Denoising and image restoration**
   - Frequency-domain diagnostics and notch filtering.
   - BM3D and Noise2Void workflows.
   - PSF generation and validation.
   - Fiji/DeconvolutionLab2 and Python Richardson-Lucy deconvolution.
   - 2D, 2.5D, and 3D comparison outputs.

3. **Segmentation**
   - **Cellpose 3 is the primary supported instance-segmentation method.**
   - U-Net training and inference for 2D and 3D 2.5D data.
   - StarDist pretrained screening, fine-tuning, inference, and validation.
   - Omnipose pretrained screening, fine-tuning, inference, and validation.
   - Additional YOLO and MicrobeJ workflows.
   - Manual instance-mask evaluation using semantic and instance-level metrics.

4. **Quantitative analysis**
   - PCA-based cell orientation and spatial standardization.
   - Morphological and fluorescence-intensity measurements.
   - Axial and radial profiles.
   - Standardized-cell maps, homogeneity analysis, and colocalization analysis.
   - Welch tests, effect sizes, multiple-testing correction, and two-way ANOVA.
   - Thesis and publication figure generation.


## Main installation

The project currently targets Python 3.10 on Windows.

```powershell
$PROJECT = "C:\Users\ra18qyqu\PycharmProjects\Pneumo_Fluor_Toolkit_PFT"

conda create --name pft python=3.10.12 --yes
conda activate pft

conda install `
    pytorch=2.5.1 `
    torchvision=0.20.1 `
    pytorch-cuda=11.8 `
    -c pytorch `
    -c nvidia `
    --yes

Set-Location -LiteralPath $PROJECT

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip check
python scripts\check_project_imports.py
```

`python -m pip install -e .` installs the standard PFT dependencies and Cellpose
3.1.1.2. Optional features can be added separately:

```powershell
# TensorFlow U-Net
python -m pip install -e ".[unet]"

# BM3D and napari
python -m pip install -e ".[bm3d,napari]"

# Development and tests
python -m pip install -e ".[dev]"
```

Use the dedicated setup document for StarDist and Omnipose rather than adding
those stacks to the main environment.

## Repository structure

```text
Pneumo_Fluor_Toolkit_PFT/
├── pyproject.toml
├── README.md
├── PFT_environment_setup.txt
├── PFT_responsibilities.txt
├── scripts/
│   ├── denoising/
│   ├── segmentation/
│   └── statistics/
├── src/
│   └── PFT/
│       ├── core_prog_parts/
│       │   ├── denoising/
│       │   ├── segmentation/
│       │   └── statistics/
│       └── napari/
├── tests/
├── models/
└── results/
```

### `scripts/`

Contains user-facing terminal commands and workflow launchers. Scripts should
parse arguments, resolve user choices, call reusable package functions, and
report output locations. Scientific processing logic should not be duplicated
here when it can be placed in `src/PFT/core_prog_parts/`.

### `src/PFT/core_prog_parts/`

Contains reusable implementation code.

- `denoising/`: filtering, Noise2Void, PSF, and deconvolution.
- `segmentation/`: model I/O, training, inference, validation, mask processing,
  and metric calculation.
- `statistics/`: PCA alignment, quantitative measurements, plots, and tests.

Shared CZI, OME-Zarr, image, path, plotting, metadata, and validation utilities
remain directly under `src/PFT/core_prog_parts/` when they serve more than one
processing category.

### `models/`

Contains trained and fine-tuned model files. Current segmentation models are
organized below:

```text
models/segmentation/<method>/<dataset>/<source_mode>/<run_name>/
```

## Project documentation

- [`PFT_environment_setup.txt`](PFT_environment_setup.txt) defines how to create,
  verify, activate, switch, and reproduce the Python environments.
- [`PFT_responsibilities.txt`](PFT_responsibilities.txt) identifies the role of
  each user-facing script and reusable core module.
- Per-sample metadata and OME-Zarr validation reports are written with the image
  outputs.
- Model training and validation directories retain split information, metrics,
  plots, and configuration records.

