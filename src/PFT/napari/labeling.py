import shutil
from pathlib import Path

import numpy as np
import tifffile as tiff
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_dilation, disk


def choose_from_list(title, options):
    if not options:
        raise RuntimeError(f"No options available for: {title}")
    print(f"\n{title}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        s = input("Choose number: ").strip()
        try:
            k = int(s)
            if 1 <= k <= len(options):
                return options[k - 1]
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def find_candidate_images(sample_dir: Path):
    preferred = [
        "image_norm16_rgb.tif",
        "image_norm16_rgb.tiff",
        "image_norm16.tif",
        "image_norm16.tiff",
        "image_raw_rgb.tif",
        "image_raw_rgb.tiff",
        "image_raw.tif",
        "image_raw.tiff",
    ]

    files = {p.name.lower(): p for p in sample_dir.glob("*.tif*")}
    candidates = []

    for name in preferred:
        p = files.get(name.lower())
        if p is not None:
            candidates.append(p)

    for p in sorted(sample_dir.glob("*.tif*")):
        if p not in candidates:
            candidates.append(p)

    return candidates


def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        x = img.astype(np.float32)
        lo = np.percentile(x, 1)
        hi = np.percentile(x, 99.8)
        x = (x - lo) / (hi - lo + 1e-8)
        x = np.clip(x, 0, 1)
        u8 = (x * 255).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=-1)

    if img.ndim == 3 and img.shape[-1] in (3, 4):
        x = img[..., :3]
        if x.dtype == np.uint8:
            return x
        xf = x.astype(np.float32)
        lo = np.percentile(xf, 1)
        hi = np.percentile(xf, 99.8)
        xf = (xf - lo) / (hi - lo + 1e-8)
        xf = np.clip(xf, 0, 1)
        return (xf * 255).astype(np.uint8)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def create_thick_yellow_outline(mask01, thickness=3):
   
    boundaries = find_boundaries(mask01.astype(bool), mode="outer")
    thick = binary_dilation(boundaries, disk(thickness))
    return thick


def save_overlay_yellow_outline(image, mask01, out_png):
    rgb = to_rgb_uint8(image)
    outline = create_thick_yellow_outline(mask01, thickness=3)

    out = rgb.copy()
    out[outline, 0] = 255
    out[outline, 1] = 255
    out[outline, 2] = 0

    tiff.imwrite(str(out_png), out)


def save_composite_triptych(image, mask01, out_png):
    import matplotlib.pyplot as plt

    rgb = to_rgb_uint8(image)
    outline = create_thick_yellow_outline(mask01, thickness=3)

    overlay = rgb.copy()
    overlay[outline, 0] = 255
    overlay[outline, 1] = 255
    overlay[outline, 2] = 0

    mask_show = mask01.astype(np.uint8)

    fig = plt.figure(figsize=(12, 4), dpi=150)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(rgb)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(overlay)
    ax2.set_title("Original + Yellow Outline")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(mask_show, cmap="gray", vmin=0, vmax=1)
    ax3.set_title("Mask (0/1)")
    ax3.axis("off")

    plt.tight_layout()
    fig.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)


def annotate_with_napari(image):
    import napari

    H, W = image.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    viewer = napari.Viewer(title="U-Net Mask Annotation (Cells=1, Background=0)")
    viewer.add_image(image, name="image")

    labels_layer = viewer.add_labels(mask, name="mask", opacity=0.45)
    labels_layer.selected_label = 1
    labels_layer.brush_size = 20

    print("\nPaint cells with label=1. Close napari to save.\n")

    napari.run()

    result = labels_layer.data.astype(np.uint8)
    result = (result > 0).astype(np.uint8)
    return result


def main():
    repo_root = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT")
    img_root = repo_root / "results" / "img"
    out_root = repo_root / "results" / "training_files" / "U-net"

    datasets = ["2d_time", "2d_wga_dapi"]
    dataset = choose_from_list("Choose dataset:", datasets)

    ds_dir = img_root / dataset
    samples = sorted([p.name for p in ds_dir.iterdir() if p.is_dir()])
    sample = choose_from_list(f"Choose sample folder in {dataset}:", samples)

    sample_dir = ds_dir / sample
    candidates = find_candidate_images(sample_dir)

    chosen_name = choose_from_list(
        "Choose image file to annotate:",
        [p.name for p in candidates],
    )
    img_path = sample_dir / chosen_name

    image = tiff.imread(str(img_path))
    mask01 = annotate_with_napari(image)

    save_dir = out_root / dataset / sample
    save_dir.mkdir(parents=True, exist_ok=True)

    out_img = save_dir / "image.tif"
    shutil.copyfile(str(img_path), str(out_img))

    out_mask = save_dir / "mask.tif"
    tiff.imwrite(str(out_mask), mask01.astype(np.uint8))

    out_mask_vis = save_dir / "mask_vis.tif"
    tiff.imwrite(str(out_mask_vis), mask01.astype(np.uint8) * 255)

    out_overlay = save_dir / "overlay_yellow_outline.png"
    save_overlay_yellow_outline(image, mask01, out_overlay)

    out_composite = save_dir / "composite_triptych.png"
    save_composite_triptych(image, mask01, out_composite)

    print("\nSaved with THICK YELLOW outlines.")
    print(f"Foreground ratio: {mask01.mean():.4%}")


if __name__ == "__main__":
    main()