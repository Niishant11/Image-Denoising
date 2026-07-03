import os


def moderate_denoise(
    input_path: str,
    output_path: str,
    strength: float = 0.45,
    clarity: float = 0.12,
    contrast: float = 0.1,
    sharpen: float = 0.18,
) -> bool:
    """
    Removes moderate noise while preserving details.
    Uses non-local means + bilateral filter + selective sharpening + blending.
    """
    if not os.path.exists(input_path):
        return False

    try:
        import cv2

        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            return False

        strength = max(0.0, min(1.0, float(strength)))
        clarity = max(0.0, min(0.5, float(clarity)))
        contrast = max(0.0, min(0.5, float(contrast)))
        sharpen = max(0.0, min(0.5, float(sharpen)))

        h_val = 3 + int(strength * 12)
        h_col = 3 + int(strength * 12)

        denoised = cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h=h_val,
            hColor=h_col,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        smooth = cv2.bilateralFilter(
            denoised,
            d=7,
            sigmaColor=60,
            sigmaSpace=60,
        )

        gaussian = cv2.GaussianBlur(smooth, (0, 0), 1.0)
        clear = cv2.addWeighted(smooth, 1.0 + clarity, gaussian, -clarity, 0)

        if contrast > 0.0:
            clear = cv2.convertScaleAbs(clear, alpha=1.0 + contrast, beta=0)

        if sharpen > 0.0:
            blur = cv2.GaussianBlur(clear, (0, 0), 1.0)
            clear = cv2.addWeighted(clear, 1.0 + sharpen, blur, -sharpen, 0)

        blend = 0.6 + (strength * 0.3)
        final = cv2.addWeighted(clear, blend, img, 1.0 - blend, 0)

        cv2.imwrite(output_path, final)
        return True
    except Exception:
        # Fallback to PIL if OpenCV is unavailable
        try:
            from PIL import Image, ImageFilter, ImageEnhance

            strength = max(0.0, min(1.0, float(strength)))
            clarity = max(0.0, min(0.5, float(clarity)))
            contrast = max(0.0, min(0.5, float(contrast)))
            sharpen = max(0.0, min(0.5, float(sharpen)))

            img = Image.open(input_path).convert("RGB")
            denoised = img.filter(ImageFilter.MedianFilter(size=3))
            smooth = denoised.filter(ImageFilter.GaussianBlur(radius=0.6 + strength))

            if contrast > 0.0:
                smooth = ImageEnhance.Contrast(smooth).enhance(1.0 + contrast)

            if clarity > 0.0:
                smooth = ImageEnhance.Sharpness(smooth).enhance(1.0 + clarity)

            if sharpen > 0.0:
                smooth = ImageEnhance.Sharpness(smooth).enhance(1.0 + sharpen)

            blend = 0.6 + (strength * 0.3)
            final = Image.blend(img, smooth, alpha=blend)
            final.save(output_path)
            return True
        except Exception:
            return False
