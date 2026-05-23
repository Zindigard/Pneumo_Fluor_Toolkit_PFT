from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.segmentation_model_io_core import (
    collect_training_pairs, extract_2d_image, extract_3d_volume, load_2d_mask, load_3d_mask,
    model_root, normalize_float32, write_json,
)

DATASET = "3d"
FAMILY = "stardist"


@dataclass
class StarDistFineTuneConfig:
    project_root: Path
    epochs: int = 100
    steps_per_epoch: int = 100
    batch_size: int = 4
    patch_size: tuple[int, int] = (256, 256)
    val_fraction: float = 0.2
    seed: int = 42
    channels: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = (0, 1) if DATASET == "2d_wga_dapi" else (0,)

    def model_dir(self) -> Path:
        return model_root(self.project_root, FAMILY, DATASET)


def _load_training_arrays(cfg: StarDistFineTuneConfig):
    pairs = collect_training_pairs(cfg.project_root, DATASET)
    X, Y = [], []
    for image_path, mask_path, sample in pairs:
        if DATASET == "3d":
            vol = extract_3d_volume(image_path, channels=cfg.channels)
            mask = load_3d_mask(mask_path)
            z_count = min(vol.shape[0], mask.shape[0])
            for z in range(z_count):
                X.append(normalize_float32(vol[z]))
                Y.append(mask[z].astype(np.uint16))
        else:
            X.append(extract_2d_image(image_path, DATASET, channels=cfg.channels))
            Y.append(load_2d_mask(mask_path).astype(np.uint16))
    if not X:
        raise RuntimeError(f"No StarDist training arrays loaded for {DATASET}")
    idx = list(range(len(X)))
    random.Random(cfg.seed).shuffle(idx)
    n_val = max(1, int(round(len(idx) * cfg.val_fraction))) if len(idx) > 1 else 0
    val_idx = set(idx[:n_val])
    X_val = [X[i] for i in idx if i in val_idx]
    Y_val = [Y[i] for i in idx if i in val_idx]
    X_train = [X[i] for i in idx if i not in val_idx]
    Y_train = [Y[i] for i in idx if i not in val_idx]
    if not X_val:
        X_val, Y_val = X_train[:1], Y_train[:1]
    return X_train, Y_train, X_val, Y_val


def finetune_stardist_3d(cfg: StarDistFineTuneConfig | None = None) -> Path:
    if cfg is None:
        cfg = StarDistFineTuneConfig(project_root=find_project_root())
    from stardist.models import Config2D, StarDist2D
    out = cfg.model_dir()
    X_train, Y_train, X_val, Y_val = _load_training_arrays(cfg)
    n_channel = 1 if np.asarray(X_train[0]).ndim == 2 else int(np.asarray(X_train[0]).shape[-1])
    conf = Config2D(n_channel_in=n_channel, train_batch_size=cfg.batch_size, train_patch_size=cfg.patch_size)
    model = StarDist2D(conf, name="model", basedir=str(out))
    write_json(out / "training_config.json", cfg)
    model.train(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
    )
    print(f"[StarDist {DATASET}] model saved: {out / 'model'}")
    return out / "model"
