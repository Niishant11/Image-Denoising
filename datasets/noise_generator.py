import numpy as np
import cv2
from typing import Tuple


def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Add additive white Gaussian noise to an image.

    Args:
        image: Input image in [0, 255], uint8.
        sigma: Standard deviation of noise.

    Returns:
        Noisy image, uint8.
    """
    if image.dtype != np.float32:
        img = image.astype(np.float32)
    else:
        img = image

    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_poisson_noise(image: np.ndarray) -> np.ndarray:
    """
    Add Poisson noise to an image.

    Args:
        image: Input image in [0, 255], uint8.

    Returns:
        Noisy image, uint8.
    """
    if image.dtype != np.float32:
        img = image.astype(np.float32)
    else:
        img = image

    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img * vals) / float(vals)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_speckle_noise(image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Add speckle noise to an image.

    Args:
        image: Input image in [0, 255], uint8.
        sigma: Noise scale.

    Returns:
        Noisy image, uint8.
    """
    if image.dtype != np.float32:
        img = image.astype(np.float32)
    else:
        img = image

    noise = np.random.randn(*img.shape).astype(np.float32) * sigma * 255.0
    noisy = img + img * noise / 255.0
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_mixed_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Example of mixed noise: Gaussian + Poisson.

    Args:
        image: Input image in [0, 255], uint8.
        sigma: Gaussian sigma.

    Returns:
        Noisy image, uint8.
    """
    g_noisy = add_gaussian_noise(image, sigma)
    p_noisy = add_poisson_noise(g_noisy)
    return p_noisy


if __name__ == "__main__":
    # Simple quick test (won't run unless you add a sample image)
    print("Noise generator module ready.")
