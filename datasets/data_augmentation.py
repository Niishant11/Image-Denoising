import random
from PIL import Image
import torchvision.transforms.functional as TF


def augment_pair(noisy_img: Image.Image, clean_img: Image.Image):
    """
    Apply synchronized random augmentation to a noisy-clean image pair.
    Includes horizontal flip, vertical flip, and 90-degree rotations.

    Args:
        noisy_img: PIL Image (noisy input)
        clean_img: PIL Image (clean target)

    Returns:
        Tuple of augmented (noisy_img, clean_img)
    """
    # Random horizontal flip
    if random.random() > 0.5:
        noisy_img = TF.hflip(noisy_img)
        clean_img = TF.hflip(clean_img)

    # Random vertical flip
    if random.random() > 0.5:
        noisy_img = TF.vflip(noisy_img)
        clean_img = TF.vflip(clean_img)

    # Random 90-degree rotation
    rot = random.choice([0, 90, 180, 270])
    if rot != 0:
        noisy_img = TF.rotate(noisy_img, rot)
        clean_img = TF.rotate(clean_img, rot)

    return noisy_img, clean_img
