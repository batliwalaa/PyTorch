import os
import shutil
from PIL import Image
import numpy as np
from scipy import io

from oxford_flowers_dataset import OxfordFlowersDataset


def _setup_test_dir(root):
    jpg_dir = os.path.join(root, "jpg")
    os.makedirs(jpg_dir, exist_ok=True)
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    img.save(os.path.join(jpg_dir, "image_00001.jpg"))
    labels = np.array([[1]])
    io.savemat(os.path.join(root, "imagelabels.mat"), {"labels": labels})


def test_single_aug_applied():
    base = os.path.join(os.path.dirname(__file__), "test_data")
    if os.path.exists(base):
        shutil.rmtree(base)
    _setup_test_dir(base)

    def raw_transform(img):
        return "RAW_APPLIED"

    def aug_transform(img):
        return "AUG_APPLIED"

    ds = OxfordFlowersDataset(
        root_dir=base,
        raw_transform=raw_transform,
        aug_transform=aug_transform,
        prescan=True,
        track_coverage=False,
    )

    # Force augmentation for deterministic test
    ds.class_aug_prob = {0: 1.0}

    image, label = ds[0]
    assert image == "AUG_APPLIED", f"expected AUG_APPLIED, got {image}"
    print("test_single_aug_applied passed")

    # Cleanup test data
    if os.path.exists(base):
        shutil.rmtree(base)

if __name__ == '__main__':
    test_single_aug_applied()
