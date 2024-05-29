import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.tiff"):
            self.image_paths.extend(
                glob.glob(os.path.join(root_dir, "**", ext), recursive=True)
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
