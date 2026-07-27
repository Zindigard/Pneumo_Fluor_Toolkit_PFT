from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from PFT.core_prog_parts.decoder_omezar import load_ome_zarr, ome_zarr_to_n2v_2d_stack
"Trains N2V models on patches extracted from OME-Zarr images, with interactive dataset and mode selection."

def project_root_from_this_file() -> Path:
    """Helper function used by this module."""
    return Path(__file__).resolve().parents[4]


def ensure_dir(p: Path) -> None:
    """Ensure that the required resource exists."""
    p.mkdir(parents=True, exist_ok=True)


def is_omezarr_store(p: Path) -> bool:
    """Helper function used by this module."""
    if not p.is_dir():
        return False
    return (p / ".zgroup").exists() or (p / "zarr.json").exists() or (p / ".zattrs").exists()


def list_zarr_samples(folder: Path) -> list[Path]:
    """
    Return only directories that look like OME-Zarr stores.
    Searches:
      - direct children
      - one level nested
    """
    if not folder.exists():
        return []

    direct = [p for p in folder.iterdir() if is_omezarr_store(p)]
    if direct:
        return sorted(direct)

    nested: list[Path] = []
    for sub in [p for p in folder.iterdir() if p.is_dir()]:
        nested.extend([p for p in sub.iterdir() if is_omezarr_store(p)])
    return sorted(nested)


def infer_axes_shape_channels(zarr_dir: Path) -> tuple[str, tuple[int, ...], int]:
    """Infer a suitable value from the available inputs."""
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
        if x.shape[-1] == 1:
            return x[None, ...]
        return x[..., None]
    if x.ndim == 4 and x.shape[-1] == 1:
        return x
    raise ValueError(f"Unexpected shape for N2V stack: {x.shape}")


def split_patches_80_20(patches: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Helper function used by this module."""
    rng = np.random.default_rng(seed)
    idx = np.arange(patches.shape[0])
    rng.shuffle(idx)
    n_train = int(np.floor(0.80 * len(idx)))
    n_train = max(1, n_train) if len(idx) >= 2 else len(idx)
    return patches[idx[:n_train]], patches[idx[n_train:]]

def choose_dataset_interactive() -> str:
    """Ask the user to choose a workflow option."""
    options = ["2d_time", "2d_wga_dapi"]
    print("Choose dataset to train:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        s = input("Enter 1 or 2: ").strip()
        if s in ("1", "2"):
            return options[int(s) - 1]
        print("Invalid input. Please enter 1 or 2.")


def choose_wga_dapi_mode(detected_c: int) -> tuple[str, list[int]]:
    """
    Returns:
      mode: "single" | "separate_both" | "joint_both"
      channels: e.g. [0], [1], [0,1]
    """
    if detected_c < 2:
        print(f"[WARN] Expected 2 channels, detected C={detected_c}. Falling back to channel 0 only.")
        return "single", [0]

    print("2d_wga_dapi training mode:")
    print("  0  = train ONLY channel 0 (DAPI/blue)  -> 1 model")
    print("  1  = train ONLY channel 1 (WGA/green)  -> 1 model")
    print("  s  = train BOTH channels separately    -> 2 models (one per channel)")
    print("  j  = train BOTH channels jointly       -> 1 model (2-channel input)")
    while True:
        s = input("Choose mode (0 / 1 / s / j): ").strip().lower()
        if s == "0":
            return "single", [0]
        if s == "1":
            return "single", [1]
        if s in ("s", "sep", "separate"):
            return "separate_both", [0, 1]
        if s in ("j", "joint", "both"):
            return "joint_both", [0, 1]
        print("Invalid input. Please enter 0, 1, s, or j.")



def load_single_channel_frames_yx(samples: list[Path], *, channel: int) -> list[np.ndarray]:
    """
    Returns list of frames as 2D arrays (Y,X).
    No normalization. Float32 cast only.
    """
    frames: list[np.ndarray] = []
    for p in samples:
        x = ome_zarr_to_n2v_2d_stack(p, channel=channel, time=None, z=None, normalize=None)
        x = ensure_nyx1(np.array(x))
        # x: (N,Y,X,1) -> frames (Y,X)
        for i in range(x.shape[0]):
            frames.append(x[i, :, :, 0].astype(np.float32, copy=False))
    return frames


def load_joint_frames_yxc(samples: list[Path], *, channels: list[int]) -> list[np.ndarray]:
    """
    Returns list of frames as (Y,X,2) arrays by stacking channel frames.
    No normalization. Float32 cast only.
    """
    if len(channels) != 2:
        raise ValueError("Joint mode supports exactly 2 channels: [0,1].")
    c0, c1 = channels

    frames: list[np.ndarray] = []
    for p in samples:
        x0 = ensure_nyx1(np.array(ome_zarr_to_n2v_2d_stack(p, channel=c0, time=None, z=None, normalize=None)))
        x1 = ensure_nyx1(np.array(ome_zarr_to_n2v_2d_stack(p, channel=c1, time=None, z=None, normalize=None)))

        n = min(x0.shape[0], x1.shape[0])
        if x0.shape[0] != x1.shape[0]:
            print(f"[WARN] {p.name}: frame count differs (C0={x0.shape[0]}, C1={x1.shape[0]}), using N={n}")

        for i in range(n):
            f0 = x0[i, :, :, 0].astype(np.float32, copy=False)
            f1 = x1[i, :, :, 0].astype(np.float32, copy=False)
            frames.append(np.stack([f0, f1], axis=-1))  # (Y,X,2)

    return frames


def _frames_to_syxc(frames: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """
    Convert a list of frames to one SYXC array.
    - frames can be (Y,X) or (Y,X,C)
    - chooses most common (Y,X) and skips others
    Returns:
      X_syxc: (S,Y,X,C)
      skipped_count
    """
    if not frames:
        raise RuntimeError("No frames provided.")

    sizes = [(int(f.shape[0]), int(f.shape[1])) for f in frames]
    target_size, _ = Counter(sizes).most_common(1)[0]

    kept: list[np.ndarray] = []
    skipped = 0

    for f in frames:
        y, x = int(f.shape[0]), int(f.shape[1])
        if (y, x) != target_size:
            skipped += 1
            continue

        if f.ndim == 2:
            kept.append(f[..., None].astype(np.float32, copy=False))  # (Y,X,1)
        elif f.ndim == 3:
            kept.append(f.astype(np.float32, copy=False))             # (Y,X,C)
        else:
            skipped += 1

    if not kept:
        raise RuntimeError("All frames were skipped due to size/dimension mismatch.")

    X_syxc = np.stack(kept, axis=0)  # (S,Y,X,C)
    return X_syxc, skipped


def generate_patches_from_frames(
    frames: list[np.ndarray],
    *,
    patch_shape=(64, 64),
    num_patches_per_img=None,
    augment=True,
    shuffle=False,
) -> tuple[np.ndarray, int]:
    """
    Uses your working pattern:
      datagen.generate_patches_from_list([SYXC], ...)

    Returns:
      patches (N,py,px,C)
      skipped_frames_count
    """
    X_syxc, skipped = _frames_to_syxc(frames)
    datagen = N2V_DataGenerator()

    patches = datagen.generate_patches_from_list(
        [X_syxc],
        num_patches_per_img=num_patches_per_img,
        shape=tuple(patch_shape),
        augment=bool(augment),
        shuffle=bool(shuffle),
    )
    patches = np.asarray(patches).astype(np.float32, copy=False)

    # Ensure (N,py,px,C)
    if patches.ndim == 3:
        patches = patches[..., None]
    if patches.ndim != 4:
        raise ValueError(f"Unexpected patches shape: {patches.shape}")

    return patches, skipped


def show_two_random_patches(X: np.ndarray, title_prefix: str, seed: int = 0, show_channel: int = 0) -> None:
    """Display a quick visual preview for inspection."""
    if X.shape[0] < 2:
        print("[WARN] Not enough patches to display 2 examples.")
        return

    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=2, replace=False)

    for k, i in enumerate(idx, start=1):
        patch = X[i]  # (py,px,C)
        c = patch.shape[-1]
        ch = int(np.clip(show_channel, 0, c - 1))
        img = patch[:, :, ch]

        plt.figure()
        plt.imshow(img, cmap="Blues")
        plt.axis("off")
        plt.title(f"{title_prefix} — patch {k} (idx={i}, show C{ch}/{c})")
        plt.show()


def train_one_model(model_name: str, models_base: Path, X: np.ndarray, X_val: np.ndarray) -> None:
    """Helper function used by this module."""
    steps = max(1, int(X.shape[0] / 128))

    config = N2VConfig(
        X,
        unet_kern_size=3,
        unet_n_first=64,
        unet_n_depth=3,
        train_steps_per_epoch=steps,
        train_epochs=25,
        train_loss="mse",
        batch_norm=True,
        train_batch_size=128,
        n2v_perc_pix=0.198,
        n2v_patch_shape=(64, 64),
        n2v_manipulator="uniform_withCP",
        n2v_neighborhood_radius=5,
        single_net_per_channel=False,
    )

    print("\n========== N2V CONFIG ==========")
    print(config)
    print("Computed train_steps_per_epoch:", steps)
    print("X channels:", X.shape[-1])
    print("================================\n")

    model = N2V(config, model_name, basedir=str(models_base))

    print("========== MODEL INFO ==========")
    print("Model name:", model_name)
    print("Model dir :", (models_base / model_name))
    try:
        km = getattr(model, "keras_model", None)
        if km is not None:
            km.summary()
        else:
            print("[WARN] model.keras_model not accessible in this N2V version.")
    except Exception as e:
        print("[WARN] Could not print keras summary:", e)
    print("================================\n")

    history = model.train(X, X_val)

    print("\n========== HISTORY KEYS ==========")
    print(sorted(list(getattr(history, "history", {}).keys())))
    print("=================================\n")


def main() -> None:
    """Helper function used by this module."""
    PATCH_SHAPE = (64, 64)
    SEED = 0

    NUM_PATCHES_PER_IMG = 8192
    AUGMENT = False
    SHUFFLE = False

    root = project_root_from_this_file()
    training_base = root / "results" / "training_files"
    models_base = root / "models"
    ensure_dir(models_base)

    dataset = choose_dataset_interactive()
    dataset_root = training_base / dataset

    all_samples = list_zarr_samples(dataset_root)
    if not all_samples:
        raise FileNotFoundError(
            f"No OME-Zarr sample directories found in: {dataset_root}\n"
            f"Tip: samples must contain .zgroup or zarr.json or .zattrs."
        )

    axes, shape, c = infer_axes_shape_channels(all_samples[0])
    print("\n========== INPUT SUMMARY ==========")
    print("Dataset:        ", dataset)
    print("Found samples:  ", len(all_samples))
    print("Example sample: ", all_samples[0].name)
    print("Detected C:     ", c)
    print("Normalization:  OFF (normalize=None)")
    print("Patch shape:    ", PATCH_SHAPE)
    print("==================================\n")

    if dataset == "2d_time":
        model_name = "n2v_2d_time_raw"

        frames = load_single_channel_frames_yx(all_samples, channel=0)

        patches, skipped_frames = generate_patches_from_frames(
            frames,
            patch_shape=PATCH_SHAPE,
            num_patches_per_img=NUM_PATCHES_PER_IMG,
            augment=AUGMENT,
            shuffle=SHUFFLE,
        )

        X, X_val = split_patches_80_20(patches, seed=SEED)

        print(f"[PATCH] created: {patches.shape} | skipped frames (size mismatch): {skipped_frames}")
        print(f"[SPLIT] train: {X.shape[0]} patches | val: {X_val.shape[0]} patches")

        show_two_random_patches(X, title_prefix=model_name, seed=SEED, show_channel=0)
        train_one_model(model_name, models_base, X, X_val)

    else:
        mode, chs = choose_wga_dapi_mode(detected_c=c)

        if mode == "single":
            ch = chs[0]
            ch_name = "DAPI_blue" if ch == 0 else "WGA_green"
            model_name = f"n2v_{dataset}_{ch_name}_raw"

            frames = load_single_channel_frames_yx(all_samples, channel=ch)
            patches, skipped_frames = generate_patches_from_frames(
                frames,
                patch_shape=PATCH_SHAPE,
                num_patches_per_img=NUM_PATCHES_PER_IMG,
                augment=AUGMENT,
                shuffle=SHUFFLE,
            )
            X, X_val = split_patches_80_20(patches, seed=SEED)

            print(f"[PATCH] created: {patches.shape} | skipped frames (size mismatch): {skipped_frames}")
            print(f"[SPLIT] train: {X.shape[0]} patches | val: {X_val.shape[0]} patches")

            show_two_random_patches(X, title_prefix=model_name, seed=SEED, show_channel=0)
            train_one_model(model_name, models_base, X, X_val)

        elif mode == "separate_both":
            for ch in chs:
                ch_name = "DAPI_blue" if ch == 0 else "WGA_green"
                model_name = f"n2v_{dataset}_{ch_name}_raw"

                frames = load_single_channel_frames_yx(all_samples, channel=ch)
                patches, skipped_frames = generate_patches_from_frames(
                    frames,
                    patch_shape=PATCH_SHAPE,
                    num_patches_per_img=NUM_PATCHES_PER_IMG,
                    augment=AUGMENT,
                    shuffle=SHUFFLE,
                )
                X, X_val = split_patches_80_20(patches, seed=SEED + ch)

                print(f"[PATCH] ({ch_name}) created: {patches.shape} | skipped frames: {skipped_frames}")
                print(f"[SPLIT] ({ch_name}) train: {X.shape[0]} | val: {X_val.shape[0]}")

                show_two_random_patches(X, title_prefix=model_name, seed=SEED + ch, show_channel=0)
                train_one_model(model_name, models_base, X, X_val)

        elif mode == "joint_both":
            model_name = f"n2v_{dataset}_JOINT_DAPI+WGA_raw"

            frames = load_joint_frames_yxc(all_samples, channels=chs)
            patches, skipped_frames = generate_patches_from_frames(
                frames,
                patch_shape=PATCH_SHAPE,
                num_patches_per_img=NUM_PATCHES_PER_IMG,
                augment=AUGMENT,
                shuffle=SHUFFLE,
            )
            X, X_val = split_patches_80_20(patches, seed=SEED)

            print(f"[PATCH] (joint) created: {patches.shape} | skipped frames: {skipped_frames}")
            print(f"[SPLIT] (joint) train: {X.shape[0]} | val: {X_val.shape[0]}")

            
            show_two_random_patches(X, title_prefix=model_name, seed=SEED, show_channel=0)
            train_one_model(model_name, models_base, X, X_val)

        else:
            raise RuntimeError(f"Unknown mode: {mode}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)