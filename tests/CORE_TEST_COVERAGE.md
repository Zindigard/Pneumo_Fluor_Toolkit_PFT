# Core test coverage

The archive preserves the previously supplied tests and adds focused tests for the existing core scripts.

## Functional test groups

- `test_common_paths_core.py`: project-root discovery, dataset aliases, result paths.
- `test_decoder_omezar_core.py`: axis normalization, indexing, CZYX conversion, N2V stack conversion, PSF config parsing.
- `test_image_and_filter_core.py`: image normalization, RGB conversion, notch filtering, flood fill, BM3D scaling, deconvolution preparation, threshold grouping.
- `test_segmentation_model_io_core.py`: mask and image discovery, 2D/3D extraction, mask loading, output roots, JSON output, TIFF/OME-Zarr label output contract.
- `test_cellpose_core_scripts.py`: all three Cellpose run cores and all three Cellpose fine-tuning cores.
- `test_stardist_core_scripts.py`: all three StarDist run cores and all three StarDist fine-tuning cores.
- `test_omnipose_core_scripts.py`: all three Omnipose run cores and all three Omnipose fine-tuning cores.
- `test_yolo_core_scripts.py`: all three YOLO run cores and all three YOLO fine-tuning cores.
- `test_unet_core_scripts.py`: 2D U-Net helpers, WGA-DAPI wrapper, 2.5D input construction, tiled prediction, dataset routing.
- `test_pca_mask_alignment_core.py`: PCA orientation, horizontal rotation, binary relabelling, prediction discovery, 2D output generation, sample selection.
- `test_core_source_inventory.py`: parses every Python file in `src/PFT/core_prog_parts` to detect syntax errors and checks that the expected processing families are present.

## Optional dependencies

Tests use lightweight mocks for Cellpose, StarDist, Omnipose, Ultralytics YOLO, and TensorFlow. This allows logic and routing to be tested without downloading models or requiring a GPU.

OME-Zarr round-trip tests from the original suite still use `pytest.importorskip` and run only when `ome-zarr` and `zarr` are installed.

## Run

From the project root:

```bash
pytest -q
```

Run only the newly added core tests:

```bash
pytest -q \
  tests/test_core_source_inventory.py \
  tests/test_common_paths_core.py \
  tests/test_decoder_omezar_core.py \
  tests/test_image_and_filter_core.py \
  tests/test_segmentation_model_io_core.py \
  tests/test_cellpose_core_scripts.py \
  tests/test_stardist_core_scripts.py \
  tests/test_omnipose_core_scripts.py \
  tests/test_yolo_core_scripts.py \
  tests/test_unet_core_scripts.py \
  tests/test_pca_mask_alignment_core.py
```
