import os


def enhance_smoothness(input_path: str, output_path: str, method: str = "bilateral") -> bool:
    """
    Apply a simple smoothing pass to reduce residual noise.

    Args:
        input_path: Path to the input image.
        output_path: Path to write the smoothed image.
        method: "bilateral" or "gaussian".

    Returns:
        True if saved successfully, False otherwise.
    """
    if not os.path.exists(input_path):
        return False

    try:
        import cv2

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            return False

        if method == "bilateral":
            # Edge-preserving smoothing
            out = cv2.bilateralFilter(img, d=9, sigmaColor=30, sigmaSpace=30)
        else:
            # Gaussian blur fallback
            out = cv2.GaussianBlur(img, (3, 3), 0)

        cv2.imwrite(output_path, out)
        return True
    except Exception:
        return False
