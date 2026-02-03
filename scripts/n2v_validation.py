import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from PFT.core_prog_parts.n2v_decoder_omezar import load_ome_zarr, ome_zarr_to_n2v_2d_stack
from PFT.core_prog_parts.Ome_Zarr import save_ome_zarr_next_to_outputs
from PFT.core_prog_parts.n2v_denoising import denoising_N2V 

REPO_ROOT = Path(__file__).resolve().parents[3]
VAL_DATA_DIR = REPO_ROOT / "results" / "training_files" / "validation"
OUTPUT_BASE = REPO_ROOT / "results" / "N2V_denoised"
MODEL_DIR = REPO_ROOT / "models"
MODEL_NAME = "PFT_N2V_2D_v1"

def validate_file(zarr_path, show_diff=True):
    zarr_path = Path(zarr_path)
    file_id = zarr_path.parent.name
    
    raw_stack = ome_zarr_to_n2v_2d_stack(zarr_path) 
    denoised_stack = denoising_N2V(raw_stack, MODEL_DIR, MODEL_NAME)
    
    # Metrics
    mse = np.mean((raw_stack.squeeze() - denoised_stack)**2)
    mae = np.mean(np.abs(raw_stack.squeeze() - denoised_stack))
    
    mid = len(raw_stack) // 2
    raw_sample = raw_stack[mid].squeeze()
    den_sample = denoised_stack[mid]
    
    if raw_sample.ndim == 3: 
        s_val = np.mean([ssim(raw_sample[..., i], den_sample[..., i], 
                             data_range=raw_sample[..., i].max()-raw_sample[..., i].min()) 
                        for i in range(raw_sample.shape[-1])])
    else: 
        s_val = ssim(raw_sample, den_sample, data_range=raw_sample.max()-raw_sample.min())

    save_path = OUTPUT_BASE / f"{file_id}²"
    save_path.mkdir(parents=True, exist_ok=True)
    save_ome_zarr_next_to_outputs(save_path, denoised_stack, meta=None)

    with open(save_path / f"{file_id}_metrics.txt", "w") as f:
        f.write(f"SSIM: {s_val:.4f}\nMasked MSE: {mse:.6f}\nMasked MAE: {mae:.6f}\n")

    if show_diff:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ch = 0 if raw_sample.ndim == 3 else ...
        ax[0].imshow(raw_sample if raw_sample.ndim==2 else raw_sample[...,0], cmap='magma')
        ax[1].imshow(den_sample if den_sample.ndim==2 else den_sample[...,0], cmap='magma')
        ax[2].imshow(raw_sample.astype(float) - den_sample.astype(float) if raw_sample.ndim==2 else (raw_sample[...,0]-den_sample[...,0]), cmap='seismic')
        
        # RGB  
        plt.figure(figsize=(8, 8), facecolor='black')
        def norm(img): return (img - img.min()) / (img.max() - img.min() + 1e-6)
        if den_sample.ndim == 3: 
            rgb = np.zeros((*den_sample.shape[:2], 3))
            rgb[..., 0], rgb[..., 1] = norm(den_sample[..., 0]), norm(den_sample[..., 1])
        else: 
            rgb = np.zeros((*den_sample.shape, 3))
            rgb[..., 2] = norm(den_sample)
        plt.imshow(rgb); plt.axis('off'); plt.title(f"Denoised RGB | SSIM: {s_val:.3f}", color='white'); plt.show()

def main():
    val_files = list(VAL_DATA_DIR.glob("*/image.ome.zarr"))
    if not val_files:
        print(f"No validation files found in {VAL_DATA_DIR}")
        return

    print(f"\n[0] Process ALL {len(val_files)} validation files")
    for i, f in enumerate(val_files, 1):
        print(f"[{i}] {f.parent.name}")
        
    choice = input("\nSelect index or 'all': ").strip().lower()
    if choice in ['0', 'all']:
        for f in val_files: validate_file(f, show_diff=False)
    else:
        validate_file(val_files[int(choice)-1], show_diff=True)

if __name__ == "__main__":
    main()