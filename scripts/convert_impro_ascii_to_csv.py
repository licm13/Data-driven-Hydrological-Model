#!/usr/bin/env python
"""
Convert IMPRO-like ASCII catchment data into the expected CSV layout used by experiments.

Usage (PowerShell):
  C:/Python314/python.exe scripts/convert_impro_ascii_to_csv.py \
    --source "F:/Github/Dataset/IMPRO_catchment_data_infotheo/iller" \
    --target "f:/Github/Data-driven-Hydrological-Model/data/raw/Iller" \
    [--config "F:/path/to/config.yaml"]

If --config is omitted, the script tries to auto-detect files.
The output directory will contain meteorology.csv, discharge.csv, and optionally a copied config.yaml.
"""
from pathlib import Path
import argparse
import shutil
from typing import Optional, Dict
import yaml
import pandas as pd

# Reuse loader internals
from src.utils.data_loader import _load_catchment_from_folder_generic


def main():
    p = argparse.ArgumentParser(description='Convert ASCII catchment data to CSV format expected by experiments')
    p.add_argument('--source', required=True, help='Path to source catchment folder (e.g., .../iller)')
    p.add_argument('--target', required=True, help='Output folder (e.g., data/raw/Iller)')
    p.add_argument('--config', default=None, help='Optional config.yaml describing file/column mappings')
    args = p.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    cfg: Optional[Dict] = None
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        # Also support config.yaml living inside the source folder
        source_cfg = source / 'config.yaml'
        if source_cfg.exists():
            with open(source_cfg, 'r') as f:
                cfg = yaml.safe_load(f)

    # Load
    met_df, q_df = _load_catchment_from_folder_generic(source, cfg)

    # Save
    met_df[['date', 'precip', 'temp', 'pet']].to_csv(target / 'meteorology.csv', index=False)
    q_df[['date', 'discharge']].to_csv(target / 'discharge.csv', index=False)

    # Copy config (optional)
    if cfg:
        with open(target / 'config.yaml', 'w') as f:
            yaml.safe_dump(cfg, f)

    print(f'Wrote: {target / "meteorology.csv"}')
    print(f'Wrote: {target / "discharge.csv"}')


if __name__ == '__main__':
    main()
