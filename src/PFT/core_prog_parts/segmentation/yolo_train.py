from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from ultralytics import YOLO

from PFT.core_prog_parts.common_paths import find_project_root


"""Interactive YOLO training workflow."""


def prompt_choice(title: str, options: list[str], default: int = 0) -> str:
    """Ask the user to choose one option from a numbered list."""
    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        tag = " (default)" if i - 1 == default else ""
        print(f"  {i}) {opt}{tag}")
    while True:
        s = input(f"Choose number [1-{len(options)}] or Enter for default: ").strip()
        if s == "":
            return options[default]
        try:
            idx = int(s) - 1
        except ValueError:
            idx = -1
        if 0 <= idx < len(options):
            return options[idx]
        print("Invalid choice.")


def prompt_bool(title: str, default: bool = True) -> bool:
    """Ask the user a yes or no question."""
    suffix = "Y/n" if default else "y/N"
    while True:
        s = input(f"{title} [{suffix}]: ").strip().lower()
        if s == "":
            return default
        if s in {"y", "yes", "1", "true"}:
            return True
        if s in {"n", "no", "0", "false"}:
            return False
        print("Please answer yes or no.")


def prompt_text(title: str, default: str = "") -> str:
    """Ask the user for a text value."""
    s = input(f"{title} [{default}]: ").strip()
    return s if s else default


def prompt_int(title: str, default: int) -> int:
    """Ask the user for an integer value."""
    while True:
        s = input(f"{title} [{default}]: ").strip()
        if s == "":
            return default
        try:
            return int(s)
        except ValueError:
            print("Please enter an integer.")


def prompt_float(title: str, default: float) -> float:
    """Ask the user for a float value."""
    while True:
        s = input(f"{title} [{default}]: ").strip()
        if s == "":
            return default
        try:
            return float(s)
        except ValueError:
            print("Please enter a number.")


def _default_model(task: str, model_size: str) -> str:
    """Return the default pretrained YOLO model name."""
    if task == "segment":
        return f"yolo11{model_size}-seg.pt"
    return f"yolo11{model_size}.pt"


@dataclass
class YOLOTrainConfig:
    """Store training settings for one YOLO run."""
    project_root: Path
    data_yaml: str
    task: str = "detect"
    model_size: str = "s"
    pretrained_model: str | None = None
    model_name: str | None = None
    epochs: int = 100
    imgsz: int = 640
    batch: int = 8
    device: str = "0"
    workers: int = 4
    patience: int = 30
    lr0: float = 0.01
    cos_lr: bool = False
    cache: bool = False
    single_cls: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 1.0
    copy_paste: float = 0.0

    def resolved_model(self) -> str:
        """Return the actual pretrained model that will be used."""
        return self.pretrained_model or _default_model(self.task, self.model_size)

    def run_name(self) -> str:
        """Return the output run name."""
        if self.model_name:
            return self.model_name
        return f"yolo_{self.task}_{Path(self.resolved_model()).stem}"

    def project_dir(self) -> Path:
        """Return the model output directory."""
        out = self.project_root / "models" / "yolo"
        out.mkdir(parents=True, exist_ok=True)
        return out


def build_train_config_interactive(project_root: Path | None = None) -> YOLOTrainConfig:
    """Collect all YOLO training choices from the terminal."""
    root = project_root or find_project_root()
    task = prompt_choice("Choose YOLO task:", ["detect", "segment"], default=0)
    model_size = prompt_choice("Choose model size:", ["n", "s", "m", "l", "x"], default=1)
    default_model = _default_model(task, model_size)
    custom_model = prompt_text("Custom pretrained model path or model name", default_model)
    run_name = prompt_text("Run name", f"yolo_{task}_{model_size}")

    return YOLOTrainConfig(
        project_root=root,
        data_yaml=prompt_text("Dataset YAML path", str(root / "data.yaml")),
        task=task,
        model_size=model_size,
        pretrained_model=custom_model,
        model_name=run_name,
        epochs=prompt_int("Epochs", 100),
        imgsz=prompt_int("Image size", 640),
        batch=prompt_int("Batch size", 8),
        device=prompt_text("Device", "0"),
        workers=prompt_int("Workers", 4),
        patience=prompt_int("Patience", 30),
        lr0=prompt_float("Initial learning rate", 0.01),
        cos_lr=prompt_bool("Use cosine learning-rate schedule", False),
        cache=prompt_bool("Cache dataset", False),
        single_cls=prompt_bool("Single-class mode", True),
        hsv_h=prompt_float("HSV-H augmentation", 0.015),
        hsv_s=prompt_float("HSV-S augmentation", 0.7),
        hsv_v=prompt_float("HSV-V augmentation", 0.4),
        degrees=prompt_float("Rotation degrees", 0.0),
        translate=prompt_float("Translation", 0.1),
        scale=prompt_float("Scale", 0.5),
        fliplr=prompt_float("Left-right flip probability", 0.5),
        mosaic=prompt_float("Mosaic probability", 1.0),
        copy_paste=prompt_float("Copy-paste probability", 0.0),
    )


def train_yolo_model(cfg: YOLOTrainConfig) -> Path:
    """Train or fine-tune a YOLO model and save outputs to disk."""
    model = YOLO(cfg.resolved_model())

    model.train(
        data=cfg.data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        patience=cfg.patience,
        lr0=cfg.lr0,
        cos_lr=cfg.cos_lr,
        cache=cfg.cache,
        single_cls=cfg.single_cls,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        fliplr=cfg.fliplr,
        mosaic=cfg.mosaic,
        copy_paste=cfg.copy_paste,
        project=str(cfg.project_dir()),
        name=cfg.run_name(),
        exist_ok=True,
    )

    run_dir = cfg.project_dir() / cfg.run_name()
    summary = asdict(cfg)
    summary["project_root"] = str(cfg.project_root)
    summary["run_dir"] = str(run_dir)
    summary["resolved_model"] = cfg.resolved_model()

    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    """Run YOLO training from terminal prompts only."""
    cfg = build_train_config_interactive()
    out_dir = train_yolo_model(cfg)
    print(f"\nSaved YOLO training outputs to: {out_dir}")


if __name__ == "__main__":
    main()
