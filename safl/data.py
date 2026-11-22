"""Explicit manifests, paired transforms, and mask-derived point/box prompts."""
import argparse
from pathlib import Path
from random import randint
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


def random_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0, 1]] = indices[:, [1, 0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0, 1]] = indices[:, [1, 0]]
    pt = indices[np.random.randint(len(indices))]
    return pt[np.newaxis, :], [point_label]


def fixed_click(mask, class_id=1):
    indices = np.argwhere(mask == class_id)
    indices[:, [0, 1]] = indices[:, [1, 0]]
    point_label = 1
    if len(indices) == 0:
        point_label = 0
        indices = np.argwhere(mask != class_id)
        indices[:, [0, 1]] = indices[:, [1, 0]]
    pt = indices[len(indices) // 2]
    return pt[np.newaxis, :], [point_label]


def random_bbox(mask, class_id=1, img_size=256):
    indices = np.argwhere(mask == class_id)
    indices[:, [0, 1]] = indices[:, [1, 0]]
    if indices.shape[0] == 0:
        return np.array([-1, -1, img_size, img_size], dtype=np.float32)

    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])

    classw_size = maxx - minx + 1
    classh_size = maxy - miny + 1

    shiftw = randint(int(0.95 * classw_size), int(1.05 * classw_size))
    shifth = randint(int(0.95 * classh_size), int(1.05 * classh_size))
    shiftx = randint(-int(0.05 * classw_size), int(0.05 * classw_size))
    shifty = randint(-int(0.05 * classh_size), int(0.05 * classh_size))

    new_centerx = (minx + maxx) // 2 + shiftx
    new_centery = (miny + maxy) // 2 + shifty

    minx = np.max([new_centerx - shiftw // 2, 0])
    maxx = np.min([new_centerx + shiftw // 2, img_size - 1])
    miny = np.max([new_centery - shifth // 2, 0])
    maxy = np.min([new_centery + shifth // 2, img_size - 1])

    return np.array([minx, miny, maxx, maxy], dtype=np.float32)


def fixed_bbox(mask, class_id=1, img_size=256):
    indices = np.argwhere(mask == class_id)
    indices[:, [0, 1]] = indices[:, [1, 0]]
    if indices.shape[0] == 0:
        return np.array([-1, -1, img_size, img_size], dtype=np.float32)
    minx = np.min(indices[:, 0])
    maxx = np.max(indices[:, 0])
    miny = np.min(indices[:, 1])
    maxy = np.max(indices[:, 1])
    return np.array([minx, miny, maxx, maxy], dtype=np.float32)


class JointTransform2D:
    """Paired image and mask transformations."""

    def __init__(self, training=False, image_size=256, low_image_size=128):
        self.training = training
        self.image_size = image_size
        self.low_image_size = low_image_size

    def __call__(self, image, mask):
        # Image-only intensity changes; geometry is shared with the mask.
        if self.training and np.random.rand() < 0.3:
            gamma = np.random.randint(10, 25) / 10.0
            image = ((image / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        if self.training and np.random.rand() < 0.5:
            image, mask = F.hflip(image), F.hflip(mask)
        if self.training and np.random.rand() < 0.2:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        if self.training and np.random.rand() < 0.2:
            size = int(self.image_size * np.random.uniform(1, 1.3))
            image = F.resize(image, (size, size), InterpolationMode.BILINEAR)
            mask = F.resize(mask, (size, size), InterpolationMode.NEAREST)
            crop = T.RandomCrop.get_params(image, (self.image_size, self.image_size))
            image, mask = F.crop(image, *crop), F.crop(mask, *crop)
        if self.training and np.random.rand() < 0.2:
            image = F.gaussian_blur(image, kernel_size=[3, 3], sigma=[0.1, 1.0])
        if self.training and np.random.rand() < 0.2:
            array = np.array(image).astype(np.float32)
            noise = np.random.normal(size=array.shape) * np.random.randint(2, 8)
            image = F.to_pil_image(np.clip(array + noise, 0, 255).astype(np.uint8))
        if self.training and np.random.rand() < 0.3:
            image = T.ColorJitter(brightness=0.25, contrast=(0.75, 1.35))(image)
        size = (self.image_size, self.image_size)
        image = F.resize(image, size, InterpolationMode.BILINEAR)
        mask = F.resize(mask, size, InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_image_size, self.low_image_size), InterpolationMode.NEAREST)
        return (F.to_tensor(image),
                torch.from_numpy(np.array(mask, dtype=np.int64)),
                torch.from_numpy(np.array(low_mask, dtype=np.int64)))


def read_manifest(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.strip().split("/")
        if (len(parts) != 4 or parts[0] != "1" or parts[3] not in ("0", "1")
                or any(not p or p in (".", "..") or "\\" in p or ":" in p for p in parts)):
            raise ValueError(f"Invalid four-field binary manifest row at line {number}")
        rows.append(tuple(parts))
    if not rows:
        raise ValueError("Manifest contains no cases")
    keys = [(row[1], row[2]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate case identifiers in manifest")
    return rows


class UltrasoundDataset(Dataset):
    """Paired grayscale images, binary masks and class labels."""

    def __init__(self, data_root, manifest, training=False, image_size=256, low_image_size=128):
        self.root = Path(data_root).resolve()
        self.rows = read_manifest(manifest)
        self.training = training
        self.image_size = image_size
        self.transform = JointTransform2D(training, image_size, low_image_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        _, subset, filename, label = self.rows[index]
        images = []
        for folder in ("Image", "Label"):
            path = (self.root / subset / folder / (filename + ".png")).resolve()
            if self.root not in path.parents:
                raise ValueError("Case path leaves data root")
            array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if array is None:
                raise FileNotFoundError(f"Cannot read {folder} PNG for manifest case {index}")
            images.append(array)
        image, mask = images
        if image.shape != mask.shape:
            raise ValueError(f"Image/mask dimensions differ at case {index}")
        mask = (mask > 0).astype(np.uint8)
        image, mask, low_mask = self.transform(image, mask)
        mask_np = mask.numpy()
        point, point_label = (random_click(mask_np) if self.training else fixed_click(mask_np))
        box = (random_bbox(mask_np, img_size=self.image_size) if self.training
               else fixed_bbox(mask_np, img_size=self.image_size))
        return {"image": image, "label": mask.unsqueeze(0), "low_mask": low_mask.unsqueeze(0),
                "pt": np.asarray(point, dtype=np.float32),
                "p_label": np.asarray(point_label, dtype=np.int64),
                "bbox": np.asarray(box, dtype=np.float32), "cls_label": int(label),
                "image_name": f"{subset}/{filename}.png"}


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def main():
    parser = argparse.ArgumentParser(description="Check paired PNG data and prompts")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    options = parser.parse_args()
    dataset = UltrasoundDataset(options.data_root, options.manifest)
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["image"].shape == (1, 256, 256)
        assert sample["low_mask"].shape == (1, 128, 128)
    print(f"Validated {len(dataset)} image/mask pairs and GT-derived prompts.")


if __name__ == "__main__":
    main()
