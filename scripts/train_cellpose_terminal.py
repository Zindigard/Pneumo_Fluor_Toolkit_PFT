
from __future__ import annotations
from pathlib import Path
from PFT.core_prog_parts.common_paths import find_project_root
from PFT.core_prog_parts.cellpose_finetune_2d_time_core import CellposeFineTune2DTimeConfig, finetune_cellpose as finetune_2d_time
from PFT.core_prog_parts.cellpose_finetune_2d_wga_dapi_core import CellposeFineTune2DWgaDapiConfig, finetune_cellpose as finetune_2d_wga_dapi
from PFT.core_prog_parts.cellpose_finetune_3d_core import CellposeFineTune3DConfig, finetune_cellpose as finetune_3d

"Use this script to run Cellpose fine-tuning. The output will be saved as OME-Zarr."

def _yes_no(prompt: str, default: bool = True) -> bool:
    suffix = 'Y/n' if default else 'y/N'
    ans = input(f'{prompt} [{suffix}]: ').strip().lower()
    if not ans:
        return default
    return ans in {'y', 'yes', '1', 'true'}


def _float_or_none(prompt: str, default: float | None) -> float | None:
    shown = 'None' if default is None else str(default)
    ans = input(f'{prompt} [{shown}]: ').strip()
    if not ans:
        return default
    if ans.lower() in {'none', 'no', 'n'}:
        return None
    return float(ans)


def _int_value(prompt: str, default: int) -> int:
    ans = input(f'{prompt} [{default}]: ').strip()
    return int(ans) if ans else default


def _float_value(prompt: str, default: float) -> float:
    ans = input(f'{prompt} [{default}]: ').strip()
    return float(ans) if ans else default


def main() -> None:
    project_root = find_project_root()
    print('\nCellpose fine-tuning launcher')
    print(f'Project root: {project_root}')
    print('\nChoose dataset:')
    print('  1) 2d_time')
    print('  2) 2d_wga_dapi')
    print('  3) 3d, slice-wise Cellpose fine-tuning')
    choice = input('Selection [1/2/3]: ').strip()

    pretrained = input('Pretrained Cellpose model [cpsam]: ').strip() or 'cpsam'
    gpu = _yes_no('Use GPU?', True)
    epochs = _int_value('Epochs', 200)
    batch = _int_value('Batch size', 4)
    lr = _float_value('Learning rate', 1e-5)
    test_fraction = _float_value('Validation fraction', 0.2)

    if choice == '1':
        cfg = CellposeFineTune2DTimeConfig(
            project_root=project_root, pretrained_model=pretrained, gpu=gpu,
            n_epochs=epochs, batch_size=batch, learning_rate=lr, test_fraction=test_fraction,
        )
        out = finetune_2d_time(cfg)
    elif choice == '2':
        cfg = CellposeFineTune2DWgaDapiConfig(
            project_root=project_root, pretrained_model=pretrained, gpu=gpu,
            n_epochs=epochs, batch_size=batch, learning_rate=lr, test_fraction=test_fraction,
        )
        out = finetune_2d_wga_dapi(cfg)
    elif choice == '3':
        cfg = CellposeFineTune3DConfig(
            project_root=project_root, pretrained_model=pretrained, gpu=gpu,
            n_epochs=epochs, batch_size=batch, learning_rate=lr, test_fraction=test_fraction,
        )
        out = finetune_3d(cfg)
    else:
        raise SystemExit('Unknown choice.')

    print(f'\nSaved Cellpose fine-tuning outputs to:\n{out}')
    print('Models are under:')
    print(project_root / 'models')


if __name__ == '__main__':
    main()
