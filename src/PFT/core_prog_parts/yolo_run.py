from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from ultralytics import YOLO

from PFT.core_prog_parts.common_paths import find_project_root


""" YOLO prediction workflow."""


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


@dataclass
class YOLORunConfig:
    """Store prediction settings for one YOLO run."""
    project_root: Path
    source: str
    weights: str
    task: str = "detect"
    save_txt: bool = True
    save_conf: bool = True
    save_crop: bool = False
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.7
    device: str = "0"
    max_det: int = 300
    agnostic_nms: bool = False
    show_boxes: bool = True
    show_labels: bool = True
    retina_masks: bool = True
    model_name: str | None = None

    def run_name(self) -> str:
        """Return the output run name."""
        if self.model_name:
            return self.model_name
        return Path(self.weights).stem

    def output_root(self) -> Path:
        """Return the output directory for raw YOLO predictions."""
        out = self.project_root / "results" / "segmentation_predictions" / "yolo" / self.task
        out.mkdir(parents=True, exist_ok=True)
        return out


def build_run_config_interactive(project_root: Path | None = None) -> YOLORunConfig:
    """Collect all YOLO prediction choices from the terminal."""
    root = project_root or find_project_root()
    task = prompt_choice("Choose YOLO task:", ["detect", "segment"], default=0)
    weights = prompt_text("Weights path", str(root / "models" / "yolo" / "best.pt"))

    return YOLORunConfig(
        project_root=root,
        source=prompt_text("Source path (image, folder, video, or glob)", "images"),
        weights=weights,
        task=task,
        save_txt=prompt_bool("Save txt labels", True),
        save_conf=prompt_bool("Save confidences", True),
        save_crop=prompt_bool("Save crops", False),
        imgsz=prompt_int("Image size", 640),
        conf=prompt_float("Confidence threshold", 0.25),
        iou=prompt_float("IoU threshold", 0.7),
        device=prompt_text("Device", "0"),
        max_det=prompt_int("Maximum detections", 300),
        agnostic_nms=prompt_bool("Agnostic NMS", False),
        show_boxes=prompt_bool("Show boxes", True),
        show_labels=prompt_bool("Show labels", True),
        retina_masks=prompt_bool("Use retina masks", True) if task == "segment" else False,
        model_name=prompt_text("Output folder name", Path(weights).stem),
    )


def run_yolo_dataset(cfg: YOLORunConfig) -> Path:
    """Run YOLO prediction and save results to disk."""
    model = YOLO(cfg.weights)

    model.predict(
        source=cfg.source,
        imgsz=cfg.imgsz,
        conf=cfg.conf,
        iou=cfg.iou,
        device=cfg.device,
        max_det=cfg.max_det,
        agnostic_nms=cfg.agnostic_nms,
        save=True,
        save_txt=cfg.save_txt,
        save_conf=cfg.save_conf,
        save_crop=cfg.save_crop,
        show_boxes=cfg.show_boxes,
        show_labels=cfg.show_labels,
        retina_masks=cfg.retina_masks if cfg.task == "segment" else False,
        project=str(cfg.output_root()),
        name=cfg.run_name(),
        exist_ok=True,
    )

    out_dir = cfg.output_root() / cfg.run_name()
    summary = asdict(cfg)
    summary["project_root"] = str(cfg.project_root)
    summary["output_dir"] = str(out_dir)

    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    """Run YOLO prediction from terminal prompts only."""
    cfg = build_run_config_interactive()
    out_dir = run_yolo_dataset(cfg)
    print(f"\nSaved YOLO predictions to: {out_dir}")


if __name__ == "__main__":
    main()
