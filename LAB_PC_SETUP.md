# FAU laboratory PC connection and repository setup

## Connect to the FAU network

1. Start the FAU VPN client.
2. Connect to `vpn.fau.de`.
3. Select **Full Tunnel**.

## Open the Mt Everest workstation

1. Press `Win + R`.
2. Enter `mstsc`.
3. Connect to `10.203.184.27`.
4. Use the account format `FAUAD\your-idm-id`.
5. Enter the normal FAU IdM password.

## Clone the repository

```powershell
New-Item -ItemType Directory -Force D:\Thesis | Out-Null
Set-Location D:\Thesis
git clone https://github.com/Zindigard/Pneumo_Fluor_Toolkit_PFT.git
Set-Location D:\Thesis\Pneumo_Fluor_Toolkit_PFT
```

The expected code locations are:

```text
D:\Thesis\Pneumo_Fluor_Toolkit_PFT\scripts
D:\Thesis\Pneumo_Fluor_Toolkit_PFT\src\PFT\core_prog_parts
```

## Create the Python environment

```powershell
conda create -n pft python=3.10 -y
conda activate pft
python -m pip install --upgrade pip setuptools wheel
Set-Location D:\Thesis\Pneumo_Fluor_Toolkit_PFT
python -m pip install -e ".[all,dev]"
```

## Verify both package roots

```powershell
python -c "import PFT; print(PFT.__file__)"
python -c "import scripts; print(scripts.__file__)"
python -c "from PFT.core_prog_parts import common_paths; print('core import OK')"
python -c "from scripts.segmentation import run_cellpose_terminal; print('script import OK')"
```

Direct script execution is also supported:

```powershell
python scripts\segmentation\run_cellpose_terminal.py
```

The installed command is:

```powershell
pft-run-cellpose
```

## Git workflow

Before working:

```powershell
Set-Location D:\Thesis\Pneumo_Fluor_Toolkit_PFT
git pull origin main
```

After changing the reorganized code:

```powershell
git status
git add scripts src pyproject.toml README.md tests
git commit -m "Reorganize processing modules"
git push origin main
```

Keep microscopy datasets, OME-Zarr stores, models, results, CZI/TIFF files, and large ZIP archives outside Git.
