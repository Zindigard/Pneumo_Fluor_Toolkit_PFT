from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
if not hasattr(np, "product"):
    np.product = np.prod  # type: ignore[attr-defined]
from tensorflow import keras
from n2v.models import N2VConfig, N2V
import tensorflow as tf
from PFT.core_prog_parts.n2v_decoder_omezar import load_ome_zarr, ome_zarr_to_n2v_2d_stack
TRAIN_DIRNAME = "training_data"
VAL_DIRNAME = "validation_data"


def project_root_from_this_file() -> Path:
    """
    Root from file location:

    """
    return Path(__file__).resolve().parents[3]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_zarr_samples(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_dir()])


def infer_axes_shape_channels(zarr_dir: Path) -> tuple[str, tuple[int, ...], int]:
    data, axes = load_ome_zarr(zarr_dir, as_numpy=False)
    shape = tuple(getattr(data, "shape", ()))
    if "c" not in axes:
        return axes, shape, 1
    c_idx = axes.index("c")
    if c_idx >= len(shape):
        return axes, shape, 1
    return axes, shape, int(shape[c_idx])

def ensure_nyx1(x: np.ndarray) -> np.ndarray:
    """Ensure stack is (N, Y, X, 1)."""
    if x.ndim == 2:
        return x[None, ..., None]
    if x.ndim == 3:
        # (N,Y,X) -> (N,Y,X,1) OR (Y,X,1) -> (1,Y,X,1)
        if x.shape[-1] == 1:
            return x[None, ...]
        return x[..., None]
    if x.ndim == 4 and x.shape[-1] == 1:
        return x
    raise ValueError(f"Unexpected shape for N2V stack: {x.shape}")


def sample_random_patches_from_stack(
    stack: np.ndarray,
    patch_shape: tuple[int, int],
    n_patches: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample random patches from a stack shaped (N, Y, X, 1).
    Returns (n_patches, py, px, 1).
    """
    stack = ensure_nyx1(stack)
    n, h, w, _ = stack.shape
    py, px = patch_shape

    if h < py or w < px:
        raise ValueError(f"Patch {py}x{px} larger than image {h}x{w}.")

    out = np.empty((n_patches, py, px, 1), dtype=stack.dtype)
    for i in range(n_patches):
        t = int(rng.integers(0, n))
        y0 = int(rng.integers(0, h - py + 1))
        x0 = int(rng.integers(0, w - px + 1))
        out[i] = stack[t, y0:y0 + py, x0:x0 + px, :]
    return out


def tile_stack_to_frames(
    stack: np.ndarray,
    tile_shape: tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    drop_incomplete: bool = True,
) -> list[np.ndarray]:
    """
    Tile each frame in a stack (N, Y, X, 1) into tiles of tile_shape.
    Returns list of frames shaped (tile_y, tile_x, 1).

    """
    stack = ensure_nyx1(stack)
    n, h, w, _ = stack.shape
    ty, tx = tile_shape

    if stride is None:
        sy, sx = ty, tx
    elif isinstance(stride, int):
        sy, sx = stride, stride
    else:
        sy, sx = stride

    tiles: list[np.ndarray] = []

    for t in range(n):
        frame = stack[t, ..., 0]  # (Y,X)

        y_starts = range(0, h, sy)
        x_starts = range(0, w, sx)

        for y0 in y_starts:
            for x0 in x_starts:
                y1, x1 = y0 + ty, x0 + tx
                if y1 <= h and x1 <= w:
                    tiles.append(frame[y0:y1, x0:x1][..., None])
                else:
                    if drop_incomplete:
                        continue
                    # pad incomplete tiles (rarely needed; optional)
                    tile = np.zeros((ty, tx), dtype=frame.dtype)
                    yy = min(ty, h - y0)
                    xx = min(tx, w - x0)
                    if yy > 0 and xx > 0:
                        tile[:yy, :xx] = frame[y0:y0 + yy, x0:x0 + xx]
                    tiles.append(tile[..., None])

    return tiles

def load_random_patches_dataset(
    samples: list[Path],
    *,
    channel: int,
    normalize: str | None,
    patch_shape: tuple[int, int],
    patches_per_file: int,
    seed: int,
) -> np.ndarray:
    """
    Loads each OME-Zarr sample and draws random patches from it.
    Returns (total_patches, py, px, 1).
    """
    rng = np.random.default_rng(seed)
    all_patches: list[np.ndarray] = []

    for p in samples:
        x = ome_zarr_to_n2v_2d_stack(p, channel=channel, time=None, z=None, normalize=normalize)  # (N,Y,X,1)
        x = ensure_nyx1(x)
        patches = sample_random_patches_from_stack(x, patch_shape=patch_shape, n_patches=patches_per_file, rng=rng)
        all_patches.append(patches)

    out = np.concatenate(all_patches, axis=0).astype(np.float32, copy=False)
    return out


def load_tiled_frames_dataset(
    samples: list[Path],
    *,
    channel: int,
    normalize: str | None,
    tile_shape: tuple[int, int],
    stride: int | tuple[int, int] | None,
    drop_incomplete: bool,
) -> list[np.ndarray]:
    """
    Loads OME-Zarr samples and returns a list of tiles (frames) shaped (tile_y, tile_x, 1).
    Use this if you want to keep all content deterministically via tiling.
    """
    frames: list[np.ndarray] = []
    for p in samples:
        x = ome_zarr_to_n2v_2d_stack(p, channel=channel, time=None, z=None, normalize=normalize)  # (N,Y,X,1)
        x = ensure_nyx1(x)
        frames.extend(tile_stack_to_frames(x, tile_shape=tile_shape, stride=stride, drop_incomplete=drop_incomplete))
    return frames

def build_callbacks(log_dir: Path, ckpt_dir: Path, save_every: int):
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)

    callbacks = [
        keras.callbacks.CSVLogger(str(log_dir / "history.csv"), append=True),
        keras.callbacks.TensorBoard(log_dir=str(log_dir / "tensorboard"), histogram_freq=0),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    class EveryNEpochs(keras.callbacks.Callback):
        def __init__(self, n: int, out_dir: Path):
            super().__init__()
            self.n = int(n)
            self.out_dir = out_dir

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.n == 0:
                path = self.out_dir / f"epoch_{epoch+1:04d}.weights.h5"
                self.model.save_weights(str(path))
                print(f"[CKPT] saved: {path}")

    callbacks.append(EveryNEpochs(save_every, ckpt_dir))
    return callbacks

def patch_modelcheckpoint_suffix_for_keras3() -> None:
    """
    Keras 3 requires: save_weights_only=True -> filepath must end with '.weights.h5'.

    """

    try:
        import keras as standalone_keras  # type: ignore
    except Exception:
        standalone_keras = None

    OrigTFModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

    class PatchedModelCheckpoint(OrigTFModelCheckpoint):
        def __init__(self, filepath, *args, **kwargs):
            save_weights_only = bool(kwargs.get("save_weights_only", False))
            fp = str(filepath)

            if save_weights_only and not fp.endswith(".weights.h5"):
                if fp.endswith(".h5"):
                    fp = fp[:-3] + ".weights.h5"   
                else:
                    fp = fp + ".weights.h5"        

            super().__init__(fp, *args, **kwargs)

    tf.keras.callbacks.ModelCheckpoint = PatchedModelCheckpoint

    if standalone_keras is not None:
        standalone_keras.callbacks.ModelCheckpoint = PatchedModelCheckpoint

    try:
        import n2v.models.n2v_standard as n2v_std
        n2v_std.ModelCheckpoint = PatchedModelCheckpoint
    except Exception:
        pass

def patch_save_weights_suffix_for_keras3() -> None:
    """
    Keras 3 requires save_weights() filename ending with `.weights.h5`.
    n2v calls save_weights('weights_last.h5') directly, so patch Model.save_weights
    to auto-fix the suffix.
    """
    import tensorflow as tf

    orig_save_weights = tf.keras.Model.save_weights

    def patched_save_weights(self, filepath, *args, **kwargs):
        fp = str(filepath)
        if not fp.endswith(".weights.h5"):
            if fp.endswith(".h5"):
                fp = fp[:-3] + ".weights.h5"
            else:
                fp = fp + ".weights.h5"
        return orig_save_weights(self, fp, *args, **kwargs)

    tf.keras.Model.save_weights = patched_save_weights

def patch_disable_care_tensorboard_image() -> None:
    """
    n2v uses csbdeep.utils.tf.CARETensorBoardImage(model=..., data=..., ...).
    On Keras 3 this callback is incompatible, so we replace it with a no-op
    that ACCEPTS arbitrary args/kwargs.
    """

    class _NoOpCallback(tf.keras.callbacks.Callback):
        def __init__(self, *args, **kwargs):
            super().__init__()

    try:
        import csbdeep.utils.tf as csb_tf
        csb_tf.CARETensorBoardImage = _NoOpCallback
    except Exception:
        pass

    try:
        import n2v.models.n2v_standard as n2v_std
        n2v_std.CARETensorBoardImage = _NoOpCallback
    except Exception:
        pass

def choose_dataset_interactive() -> str:
    options = ["2d_time", "2d_wga_dapi"]
    print("Choose dataset to train:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        s = input("Enter 1 or 2: ").strip()
        if s in ("1", "2"):
            return options[int(s) - 1]
        print("Invalid input. Please enter 1 or 2.")


def choose_channels_for_wga_dapi(detected_c: int) -> list[int]:
    """
    For 2d_wga_dapi, allow choosing:
      - channel 0 (blue=DAPI)
      - channel 1 (green=WGA)
      - both

    """
    if detected_c < 2:
        print(f"[WARN] Expected 2 channels, detected C={detected_c}. Will train only channel 0.")
        return [0]

    print("2d_wga_dapi channel selection:")
    print("  0 = blue  (DAPI)")
    print("  1 = green (WGA)")
    print("  b = both (train two models)")

    while True:
        s = input("Choose channel (0 / 1 / b): ").strip().lower()
        if s == "0":
            return [0]
        if s == "1":
            return [1]
        if s in ("b", "both"):
            return [0, 1]
        print("Invalid input. Please enter 0, 1, or b.")


def train_one_model(
    *,
    model_name: str,
    models_base: Path,
    logs_base: Path,
    X: np.ndarray,
    X_val: np.ndarray,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    patch_shape: tuple[int, int],
    learning_rate: float,
    loss: str,
    save_every: int,
) -> None:
    # --- N2V config ---
    config = N2VConfig(
        X,
        train_epochs=int(epochs),
        train_steps_per_epoch=int(steps_per_epoch),
        train_batch_size=int(batch_size),
        n2v_patch_shape=tuple(patch_shape),
        train_learning_rate=float(learning_rate),
        train_loss=str(loss),  # "mse" or "mae"
    )

    if hasattr(config, "train_checkpoint"):
        config.train_checkpoint = "weights_best.weights.h5"
    if hasattr(config, "train_checkpoint_best"):
        config.train_checkpoint_best = "weights_best.weights.h5"
    if hasattr(config, "train_checkpoint_last"):
        config.train_checkpoint_last = "weights_last.weights.h5"
    if hasattr(config, "train_checkpoint_period"):
        config.train_checkpoint_period = "weights_epoch_{epoch:04d}.weights.h5"

    model = N2V(config, model_name, basedir=str(models_base))

    patch_modelcheckpoint_suffix_for_keras3()
    patch_save_weights_suffix_for_keras3()
    patch_disable_care_tensorboard_image()
    model.train(X, X_val)



def main() -> None:
    EPOCHS = 100
    STEPS_PER_EPOCH = 200
    BATCH_SIZE = 128
    PATCH_SHAPE = (64, 64)
    LR = 1e-4
    LOSS = "mse"              # "mse" or "mae"
    SAVE_EVERY = 5
    NORMALIZE = "percentile"  # or None

    # sampling strategy
    # "random_patches" 
    # "tiling"         
    SAMPLING_MODE = "random_patches"   

    # random_patches mode 
    PATCHES_PER_FILE_TRAIN = 256
    PATCHES_PER_FILE_VAL = 64
    RANDOM_SEED = 0

    # tiling mode 
    TILE_SHAPE = (1024, 1024)          # (Y, X)
    TILE_STRIDE = None                 # None -> non-overlapping; int/(sy,sx) for overlap
    DROP_INCOMPLETE_TILES = True

    root = project_root_from_this_file()

    training_base = root / "results" / "training_files"
    logs_base = root / "results" / "N2V_denoised" / "logs"
    models_base = root / "models"

    ensure_dir(logs_base)
    ensure_dir(models_base)

    dataset = choose_dataset_interactive()
    dataset_root = training_base / dataset
    train_dir = dataset_root / TRAIN_DIRNAME
    val_dir = dataset_root / VAL_DIRNAME

    train_samples = list_zarr_samples(train_dir)
    val_samples = list_zarr_samples(val_dir)

    if not train_samples:
        raise FileNotFoundError(f"No training samples found in: {train_dir}")
    if not val_samples:
        raise FileNotFoundError(f"No validation samples found in: {val_dir}")

    axes, shape, c = infer_axes_shape_channels(train_samples[0])
    print(f"[INFO] Example store: {train_samples[0].name} axes='{axes}' shape={shape} -> C={c}")
    print(f"[INFO] Sampling mode: {SAMPLING_MODE}")

    if dataset == "2d_time":
        channels = [0]
        if c != 1:
            print(f"[WARN] Expected 1 channel for 2d_time, detected C={c}. Still using channel 0.")
    else:
        channels = choose_channels_for_wga_dapi(detected_c=c)

    for ch in channels:
        ch_name = (
            "blue"
            if dataset == "2d_time"
            else ("blue_DAPI" if ch == 0 else "green_WGA")
        )

        print(f"\n=== Training {dataset} | channel {ch} ({ch_name}) ===")

        model_name = f"n2v_{dataset}_{ch_name}_p{PATCH_SHAPE[0]}x{PATCH_SHAPE[1]}"

        print(f"[INFO] Logs:   {logs_base / model_name}")
        print(f"[INFO] Model:  {models_base / model_name}")

        if SAMPLING_MODE == "random_patches":
            X = load_random_patches_dataset(
                train_samples,
                channel=ch,
                normalize=NORMALIZE,
                patch_shape=PATCH_SHAPE,
                patches_per_file=PATCHES_PER_FILE_TRAIN,
                seed=RANDOM_SEED,
            )
            X_val = load_random_patches_dataset(
                val_samples,
                channel=ch,
                normalize=NORMALIZE,
                patch_shape=PATCH_SHAPE,
                patches_per_file=PATCHES_PER_FILE_VAL,
                seed=RANDOM_SEED + 1,
            )

        elif SAMPLING_MODE == "tiling":
            frames_train = load_tiled_frames_dataset(
                train_samples,
                channel=ch,
                normalize=NORMALIZE,
                tile_shape=TILE_SHAPE,
                stride=TILE_STRIDE,
                drop_incomplete=DROP_INCOMPLETE_TILES,
            )
            frames_val = load_tiled_frames_dataset(
                val_samples,
                channel=ch,
                normalize=NORMALIZE,
                tile_shape=TILE_SHAPE,
                stride=TILE_STRIDE,
                drop_incomplete=DROP_INCOMPLETE_TILES,
            )

            if not frames_train:
                raise RuntimeError("Tiling produced 0 training frames. Check TILE_SHAPE/STRIDE and input sizes.")
            if not frames_val:
                raise RuntimeError("Tiling produced 0 validation frames. Check TILE_SHAPE/STRIDE and input sizes.")

            X = np.stack(frames_train, axis=0).astype(np.float32, copy=False)  # (Ntiles, Ty, Tx, 1)
            X_val = np.stack(frames_val, axis=0).astype(np.float32, copy=False)

        else:
            raise ValueError("SAMPLING_MODE must be 'random_patches' or 'tiling'")

        print(f"[INFO] X: {X.shape} | X_val: {X_val.shape} | dtype: {X.dtype}")

        train_one_model(
            model_name=model_name,
            models_base=models_base,
            logs_base=logs_base,
            X=X,
            X_val=X_val,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            batch_size=BATCH_SIZE,
            patch_shape=PATCH_SHAPE,
            learning_rate=LR,
            loss=LOSS,
            save_every=SAVE_EVERY,
        )

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)