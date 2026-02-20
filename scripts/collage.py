from __future__ import annotations
from pathlib import Path
from typing import Optional
from PIL import Image, ImageOps, ImageDraw, ImageFont

"""Script to create collages of preview images from the datasets for visual comparison."""

BASE_DIR = Path(r"D:\Thesis\Pneumo_Fluor_Toolkit_PFT\results\img\2d_wga_dapi")
OUT_DIR = BASE_DIR / "collages"          
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP_KEYS = ["noNHS", "THY"]            

COLS = 2
PAD = 18
BG_TILE = (0, 0, 0)
BG_CANVAS = (255, 255, 255)


def pick_preview(folder: Path) -> Optional[Path]:
    """Prefer preview.png; then any image."""
    candidates = [
        folder / "preview.png",
        folder / "Preview.png",
    ]
    for c in candidates:
        if c.exists():
            return c

    for pat in ("preview*.png", "Preview*.png", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def to_exact_cell(img: Image.Image, cell_size: tuple[int, int], bg=BG_TILE) -> Image.Image:
    fitted = ImageOps.contain(img, cell_size)
    canvas = Image.new("RGB", cell_size, bg)
    x = (cell_size[0] - fitted.width) // 2
    y = (cell_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas



def make_collage(images: list[Image.Image], cols: int, pad: int) -> Image.Image:
    rows = (len(images) + cols - 1) // cols

    cell_w = min(im.width for im in images)
    cell_h = min(im.height for im in images)
    cell_size = (cell_w, cell_h)

    tiles = [to_exact_cell(im, cell_size) for im in images]

    out_w = cols * cell_w + (cols + 1) * pad
    out_h = rows * cell_h + (rows + 1) * pad
    canvas = Image.new("RGB", (out_w, out_h), BG_CANVAS)

    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        canvas.paste(tile, (x, y))

    return canvas


def build_group_collage(group_key: str) -> Optional[Path]:
    folders = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])

    items: list[tuple[str, Path]] = []
    for f in folders:
        if group_key.lower() not in f.name.lower():
            continue
        img_path = pick_preview(f)
        if img_path is None:
            continue
        items.append((f.name, img_path))

    if not items:
        print(f"[skip] No folders matched '{group_key}'.")
        return None

    imgs: list[Image.Image] = []
    for folder_name, img_path in items:
        im = load_rgb(img_path)
        imgs.append(im)

    collage = make_collage(imgs, cols=COLS, pad=PAD)
    out_path = OUT_DIR / f"collage_original_{group_key}.png"
    collage.save(out_path, dpi=(300, 300))

    print(f"[ok] Saved: {out_path}")
    print("     Used:")
    for folder_name, img_path in items:
        print(f"      - {folder_name} -> {img_path}")
    return out_path


def main() -> None:
    if not BASE_DIR.exists():
        raise FileNotFoundError(BASE_DIR)

    for key in GROUP_KEYS:
        build_group_collage(key)


if __name__ == "__main__":
    main()

