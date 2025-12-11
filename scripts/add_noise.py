import os
import sys

# Add project root to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

import cv2
import numpy as np
from tqdm import tqdm

from datasets.noise_generator import (
    add_gaussian_noise,
    add_poisson_noise,
    add_speckle_noise,
    add_mixed_noise,
)



def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def process_split(root_dir: str, split: str = "train"):
    """
    Create noisy versions of images from data/raw/<split> into:
      data/noisy/gaussian/<split>/
      data/noisy/nongaussian/<split>/
      data/noisy/mixed/<split>/
    """
    clean_dir = os.path.join(root_dir, "data", "raw", split)
    out_gauss = os.path.join(root_dir, "data", "noisy", "gaussian", split)
    out_nongauss = os.path.join(root_dir, "data", "noisy", "nongaussian", split)
    out_mixed = os.path.join(root_dir, "data", "noisy", "mixed", split)

    ensure_dir(out_gauss)
    ensure_dir(out_nongauss)
    ensure_dir(out_mixed)

    if not os.path.isdir(clean_dir):
        raise FileNotFoundError(f"Clean directory not found: {clean_dir}")

    files = [
        f
        for f in os.listdir(clean_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ]

    if len(files) == 0:
        raise RuntimeError(f"No image files found in {clean_dir}")

    print(f"[{split}] Found {len(files)} clean images. Generating noisy versions...")

    for fname in tqdm(files, desc=f"Processing {split}"):
        fpath = os.path.join(clean_dir, fname)
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Gaussian noise (example uses sigma=25) ---
        g_noisy = add_gaussian_noise(img, sigma=25.0)
        g_noisy_bgr = cv2.cvtColor(g_noisy, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_gauss, fname), g_noisy_bgr)

        # --- Non-Gaussian noise example (Poisson) ---
        p_noisy = add_poisson_noise(img)
        p_noisy_bgr = cv2.cvtColor(p_noisy, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_nongauss, fname), p_noisy_bgr)

        # --- Mixed noise ---
        m_noisy = add_mixed_noise(img, sigma=25.0)
        m_noisy_bgr = cv2.cvtColor(m_noisy, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_mixed, fname), m_noisy_bgr)

    print(f"[{split}] Done generating noisy images.")


if __name__ == "__main__":
    # Assume root_dir is project root "."
    root_dir = "."

    for split in ["train", "val", "test"]:
        process_split(root_dir, split=split)

    print("All splits processed. Noisy datasets are ready.")
