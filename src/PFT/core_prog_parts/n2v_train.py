import random
import shutil
import numpy as np
from pathlib import Path
from n2v.models import N2VConfig, N2V
from PFT.core_prog_parts.n2v_decoder_omezar import ome_zarr_to_n2v_2d_stack

REPO_ROOT = Path(__file__).resolve().parents[3] 
BASE_DATA = REPO_ROOT / "results" / "training_files"
TRAIN_DIR = BASE_DATA / "train"
VAL_DIR = BASE_DATA / "validation"
MODELS_DIR = REPO_ROOT / "models"

def run_multifile_training(model_name="PFT_N2V_2D_v1", epochs=25):
    all_zarrs = [f for f in BASE_DATA.glob("*/image.ome.zarr") 
                 if "train" not in str(f) and "validation" not in str(f)]
    
    if all_zarrs:
        print(f"Organizing {len(all_zarrs)} files into train/validation folders...")
        random.shuffle(all_zarrs)
        split_idx = int(len(all_zarrs) * 0.6)
        
        for i, f in enumerate(all_zarrs):
            target_sub = TRAIN_DIR if i < split_idx else VAL_DIR
            target_sub.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f.parent), str(target_sub / f.parent.name))

    train_files = list(TRAIN_DIR.glob("*/image.ome.zarr"))
    val_files = list(VAL_DIR.glob("*/image.ome.zarr"))

    def load_stack(file_list):
        return np.concatenate([ome_zarr_to_n2v_2d_stack(f) for f in file_list], axis=0)

    print(f"Loading {len(train_files)} training and {len(val_files)} validation stacks...")
    X = load_stack(train_files)
    X_val = load_stack(val_files)

    config = N2VConfig(X, unet_kern_size=3, train_steps_per_epoch=100, 
                        train_epochs=epochs, train_loss='mse', batch_norm=True, 
                        train_batch_size=128, n2v_patch_shape=(64, 64))

    model = N2V(config, model_name, basedir=str(MODELS_DIR))
    model.train(X, X_val)

if __name__ == "__main__":
    run_multifile_training()