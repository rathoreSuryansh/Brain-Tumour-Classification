"""Shared configuration for all four experiments.

Every model is trained and evaluated under exactly these settings so that
differences in the results can be attributed to the architecture.

Paths can be overridden with environment variables, so the same code runs on
Kaggle (default) and on a local machine:

    BRAIN_TUMOR_DATA     folder that contains Training/ and Testing/
    BRAIN_TUMOR_OUT      where model weights (.pth) are written
    BRAIN_TUMOR_RESULTS  where history / metrics / test outputs are written
    BRAIN_TUMOR_PRETRAINED  "0" to build ResNet18 / ViT without ImageNet weights
"""
import os
import random
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- paths -----------------------------------------------------------------
BASE_DIR = os.environ.get(
    "BRAIN_TUMOR_DATA",
    "/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset",
)
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
TEST_DIR = os.path.join(BASE_DIR, "Testing")

OUTPUT_DIR = Path(os.environ.get("BRAIN_TUMOR_OUT", REPO_ROOT / "outputs"))
RESULTS_DIR = Path(os.environ.get("BRAIN_TUMOR_RESULTS", REPO_ROOT / "results"))
FIGURES_DIR = RESULTS_DIR / "figures"

# ---- experiment protocol (identical for all models) -------------------------
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
BATCH_SIZE = 32
VAL_FRACTION = 0.1      # held out from Training/; Testing/ is never touched until the end
NUM_WORKERS = 2
SEED = 42
PRETRAINED = os.environ.get("BRAIN_TUMOR_PRETRAINED", "1") != "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG the experiments use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
