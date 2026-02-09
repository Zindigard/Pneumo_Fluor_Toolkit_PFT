from __future__ import annotations
import inspect
import sys
from pathlib import Path
from typing import Iterable
import numpy as np
from tensorflow import keras
from n2v.models import N2VConfig, N2V
from .n2v_decoder_omezar import load_ome_zarr, ome_zarr_to_n2v_2d_stack


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


def load_concat_stack(zarr_dirs: Iterable[Path], channel: int, normalize: str | None) -> np.ndarray:
    stacks: list[np.ndarray] = []
    for z in zarr_dirs:
        x = ome_zarr_to_n2v_2d_stack(z, channel=channel, time=None, z=None, normalize=normalize)
        if x.ndim != 4 or x.shape[-1] != 1:
            raise ValueError(f"Decoder returned unexpected shape {x.shape} for {z}")
        stacks.append(x)

    if not stacks:
        raise ValueError("No stacks loaded (empty dataset).")

    return np.concatenate(stacks, axis=0).astype(np.float32, copy=False)


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
      - channel 0 (green=WGA)
      - channel 1 (blue=DAPI)
      - both
    """
    if detected_c < 2:
        print(f"[WARN] Expected 2 channels, detected C={detected_c}. Will train only channel 0.")
        return [0]

    print("2d_wga_dapi channel selection:")
    print("  0 = green (WGA)")
    print("  1 = blue  (DAPI)")
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
    config = N2VConfig(
        X,
        train_epochs=int(epochs),
        train_steps_per_epoch=int(steps_per_epoch),
        train_batch_size=int(batch_size),
        n2v_patch_shape=tuple(patch_shape),
        train_learning_rate=float(learning_rate),
        train_loss=str(loss),  # "mse" or "mae"
    )

    model = N2V(config, model_name, basedir=str(models_base))

    log_dir = logs_base / model_name
    ckpt_dir = models_base / model_name / "checkpoints"
    callbacks = build_callbacks(log_dir=log_dir, ckpt_dir=ckpt_dir, save_every=save_every)

    sig = inspect.signature(model.train)
    if "callbacks" in sig.parameters:
        model.train(X, X_val, callbacks=callbacks)
    else:
        try:
            model.train(X, X_val, keras_callbacks=callbacks)
        except TypeError:
            print("[WARN] Your n2v version does not accept callbacks; training without callbacks.")
            model.train(X, X_val)


def main() -> None:
    # settings
    EPOCHS = 100
    STEPS_PER_EPOCH = 200
    BATCH_SIZE = 128
    PATCH_SHAPE = (64, 64)
    LR = 1e-4
    LOSS = "mse"              # "mse" or "mae"
    SAVE_EVERY = 5
    NORMALIZE = "percentile"  # or None
   

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
            else ("green_WGA" if ch == 0 else "blue_DAPI")
        )

        print(f"\n=== Training {dataset} | channel {ch} ({ch_name}) ===")

        X = load_concat_stack(train_samples, channel=ch, normalize=NORMALIZE)
        X_val = load_concat_stack(val_samples, channel=ch, normalize=NORMALIZE)

        model_name = f"n2v_{dataset}_{ch_name}_p{PATCH_SHAPE[0]}x{PATCH_SHAPE[1]}"

        print(f"[INFO] Logs:   {logs_base / model_name}")
        print(f"[INFO] Model:  {models_base / model_name}")
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
