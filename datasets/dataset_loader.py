import os
from typing import Optional, Callable, Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class BSD500Dataset(Dataset):
    """
    PyTorch Dataset for BSD500-style image denoising.

    It loads *paired* clean and noisy images from disk.

    Expected folder structure (relative to project root):

        data/
          raw/
            train/
            val/
            test/
          noisy/
            gaussian/
              train/
              val/
              test/
            nongaussian/
              train/
              val/
              test/
            mixed/
              train/
              val/
              test/

    By default:
      - clean images are read from:   data/raw/<split>/
      - noisy images are read from:   data/noisy/<noise_type>/<split>/
      - files are matched by filename (e.g., 123.png in both folders)
    """

    def __init__(
        self,
        root_dir: str = ".",
        split: str = "train",
        noise_type: str = "gaussian",
        clean_subdir: str = "data/raw",
        noisy_subdir: str = "data/noisy",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        patch_size: Optional[int] = None,   # 🔹 NEW: resize/crop to fixed size
    ):
        """
        Args:
            root_dir: Project root directory.
            split: One of ["train", "val", "test"].
            noise_type: One of ["gaussian", "nongaussian", "mixed"].
            clean_subdir: Relative path to clean images base folder.
            noisy_subdir: Relative path to noisy images base folder.
            transform: Transform applied to noisy images (input to model).
            target_transform: Transform applied to clean images (target).
            patch_size: If given, images are resized to (patch_size, patch_size).
        """
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        assert noise_type in ["gaussian", "nongaussian", "mixed"], (
            f"Invalid noise_type: {noise_type}"
        )

        self.root_dir = root_dir
        self.split = split
        self.noise_type = noise_type
        self.patch_size = patch_size

        # Build full paths
        self.clean_dir = os.path.join(root_dir, clean_subdir, split)
        self.noisy_dir = os.path.join(root_dir, noisy_subdir, noise_type, split)

        if not os.path.isdir(self.clean_dir):
            raise FileNotFoundError(f"Clean directory not found: {self.clean_dir}")

        if not os.path.isdir(self.noisy_dir):
            raise FileNotFoundError(f"Noisy directory not found: {self.noisy_dir}")

        # List of clean filenames
        clean_files = sorted(
            [
                f
                for f in os.listdir(self.clean_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            ]
        )

        if len(clean_files) == 0:
            raise RuntimeError(f"No image files found in clean directory: {self.clean_dir}")

        # Keep only files that exist in BOTH clean and noisy dirs
        self.samples: List[Tuple[str, str]] = []
        for fname in clean_files:
            clean_path = os.path.join(self.clean_dir, fname)
            noisy_path = os.path.join(self.noisy_dir, fname)
            if os.path.isfile(noisy_path):
                self.samples.append((noisy_path, clean_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No matching noisy-clean pairs found between:\n"
                f"  clean_dir: {self.clean_dir}\n"
                f"  noisy_dir: {self.noisy_dir}"
            )

        # Default transforms (if none provided)
        self.transform = transform
        self.target_transform = target_transform

        if self.transform is None or self.target_transform is None:
            # 🔹 If patch_size is given, resize all images to (patch_size, patch_size)
            if self.patch_size is not None:
                default_tf = T.Compose(
                    [
                        T.Resize((self.patch_size, self.patch_size)),
                        T.ToTensor(),
                    ]
                )
            else:
                # fallback: just ToTensor (no resizing)
                default_tf = T.ToTensor()

            if self.transform is None:
                self.transform = default_tf
            if self.target_transform is None:
                self.target_transform = default_tf

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_path, clean_path = self.samples[idx]

        # Open images
        noisy_img = Image.open(noisy_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        # Apply transforms
        if self.transform:
            noisy_tensor = self.transform(noisy_img)
        else:
            noisy_tensor = T.ToTensor()(noisy_img)

        if self.target_transform:
            clean_tensor = self.target_transform(clean_img)
        else:
            clean_tensor = T.ToTensor()(clean_img)

        return noisy_tensor, clean_tensor


def get_bsd500_dataloader(
    root_dir: str = ".",
    split: str = "train",
    noise_type: str = "gaussian",
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    patch_size: Optional[int] = None,  # 🔹 NEW: pass patch_size to dataset
) -> DataLoader:
    """
    Convenience function to get a DataLoader for BSD500Dataset.

    Example:
        train_loader = get_bsd500_dataloader(
            root_dir=".",
            split="train",
            noise_type="gaussian",
            batch_size=16,
            patch_size=128,
        )
    """
    dataset = BSD500Dataset(
        root_dir=root_dir,
        split=split,
        noise_type=noise_type,
        patch_size=patch_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == "train" else False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


if __name__ == "__main__":
    # Simple test for debugging
    dataset = BSD500Dataset(
        root_dir=".",
        split="train",
        noise_type="gaussian",
        patch_size=128,
    )
    print(f"Dataset size: {len(dataset)}")

    noisy, clean = dataset[0]
    print("Noisy shape:", noisy.shape)
    print("Clean shape:", clean.shape)
