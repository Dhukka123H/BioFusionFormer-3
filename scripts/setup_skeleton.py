"""
Phase 0 – Project skeleton creator.
Creates all required directories and stub files.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent  # BioFusionFormer/

DIRS = [
    # raw data
    "data/raw/target_wt/mrna",
    "data/raw/target_wt/mirna",
    "data/raw/target_wt/methylation",
    "data/raw/target_wt/clinical",
    "data/raw/reactome",
    # interim / processed / splits
    "data/interim/gdc_facets",
    "data/processed",
    "data/splits",
    # models
    "models/pretrained",
    "models/encoders",
    "models/checkpoints",
    # source tree
    "src/data",
    "src/preprocessing",
    "src/pathways",
    "src/encoders",
    "src/transformer",
    "src/fusion",
    "src/heads",
    "src/explainability",
    "src/evaluation",
    # project
    "configs",
    "notebooks",
    "scripts",
    "experiments",
    "results",
    "figures",
    "logs",
]

for d in DIRS:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
    # add __init__.py to src sub-packages
    if d.startswith("src/"):
        init = ROOT / d / "__init__.py"
        if not init.exists():
            init.touch()

# root src __init__
(ROOT / "src/__init__.py").touch()

print("[OK] All directories created.")
print(f"  Root: {ROOT.resolve()}")
