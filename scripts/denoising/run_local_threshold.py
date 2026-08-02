r"""
Activate the project environment first:

    conda activate pft

1. Command-line options


    python scripts\denoising\run_local_threshold.py --help

Available datasets:

    2d_time       HADA time-course dataset
    2d_wga_dapi   Two-channel WGA-DAPI dataset

Available processing modes:

    one       Process one selected image
    curated   Process a representative subset
    all       Process the complete dataset


2. Examples
Test one HADA image using the selected 97.5th percentile:

    python scripts\denoising\run_local_threshold.py `
        --dataset 2d_time `
        --mode one `
        --image-index 0 `
        --non-interactive `
        --high-percentile 97.5

Test one WGA-DAPI image using the selected 98.0th percentile:

    python scripts\denoising\run_local_threshold.py `
        --dataset 2d_wga_dapi `
        --mode one `
        --image-index 0 `
        --non-interactive `
        --high-percentile 98.0

The test output contains:

    image.ome.zarr
    comparison_display_normalized_original_vs_filtered_<CHANNEL>.png
    preview_display_normalized_original_<CHANNEL>.png
    preview_display_normalized_filtered_<CHANNEL>.png
    mask_keep_<CHANNEL>.png
    local_threshold_metrics.csv

 python scripts\denoising\run_local_threshold.py `
     --dataset 2d_time `
     --mode all `
     --non-interactive `
     --high-percentile 97.5


    python scripts\denoising\run_local_threshold.py `
        --dataset 2d_wga_dapi `
        --mode all `
        --non-interactive `
        --high-percentile 98.0


Validate the complete HADA result:

    python scripts\denoising\check_local_threshold_intensity.py `
        --dataset 2d_time `
        --high-percentile 97.5

Validate the complete WGA-DAPI result:

    python scripts\denoising\check_local_threshold_intensity.py `
        --dataset 2d_wga_dapi `
        --high-percentile 98.0

Calculated metrics include:

    neighbour_correlation_mean
    neighbour_correlation_maximum
    fano_factor_mean
    fano_factor_maximum
    fft_peak_score_mean
    fft_peak_score_maximum

Calculate HADA metrics:

    python scripts\denoising\FFT.py `
        --dataset 2d_time `
        --high-percentile 97.5 `
        --no-previews

Calculate WGA-DAPI metrics:

    python scripts\denoising\FFT.py `
        --dataset 2d_wga_dapi `
        --high-percentile 98.0 `
        --no-previews

"""

from PFT.core_prog_parts.denoising.denoiser import main


if __name__ == "__main__":
    main()