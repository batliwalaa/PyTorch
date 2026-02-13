import os
import logging
import csv
import re
import random
import json

from torch.utils.data import Dataset
from PIL import Image
from scipy import io
from datetime import datetime

from augmentation_coverage_tracker import AugmentationCoverageTracker
from hashing import hash_file, compute_dataset_hash

logging.basicConfig(level=logging.WARNING)


class OxfordFlowersDataset(Dataset):
    """
    Leak-safe Oxford 102 Flowers dataset.

    Design rules:
    - Offline augmentation is EXPLICIT.
    - Validation/test NEVER see augmented data.
    - Split is applied before sampling.
    """

    def __init__(
        self,
        root_dir: str,
        raw_transform,
        aug_transform=None,
        split: str | None = None,
        use_offline_aug: bool = False,
        offline_aug_version: str | None = None,
        prescan: bool = True,
        strict: bool = False,
        min_size: int = 50,
        track_coverage: bool = False,
    ):
        self.root_dir = root_dir
        self.raw_transform = raw_transform
        self.aug_transform = aug_transform
        self.use_augmentation = aug_transform is not None
        self.split = split
        self.strict = strict
        self.min_size = min_size
        self.track_coverage = track_coverage
        self.use_offline_aug = use_offline_aug

        # -----------------------
        # Load labels
        # -----------------------
        labels_path = os.path.join(root_dir, "imagelabels.mat")
        labels_mat = io.loadmat(labels_path)
        self.labels = labels_mat["labels"][0] - 1

        self.num_classes = len(set(self.labels))

        # -----------------------
        # Select image directory
        # -----------------------
        self.augmentation_version = None
        self.provenance = {}

        if self.use_offline_aug:
            if split != "train":
                raise RuntimeError("Offline augmentation is allowed ONLY for training")

            aug_root = os.path.join(root_dir, "augmented")
            if not os.path.exists(aug_root):
                raise RuntimeError("Offline augmentation requested but augmented/ not found")

            version = offline_aug_version or sorted(os.listdir(aug_root))[-1]
            self.augmentation_version = version
            self.images_dir = os.path.join(aug_root, version, "train", "images")
            self._load_provenance()

            self.use_augmentation = False
            self.aug_transform = None
        else:
            self.images_dir = os.path.join(root_dir, "jpg")

        # -----------------------
        # Prescan
        # -----------------------
        self.image_files = sorted(os.listdir(self.images_dir))
        self.valid_indices = list(range(len(self.image_files)))
        self.image_hashes = {}

        if prescan:
            self._prescan_dataset()

        # -----------------------
        # Apply split EARLY
        # -----------------------
        if self.split:
            self._apply_split(self.split)

        self.num_samples = len(self.valid_indices)
        self.coverage = AugmentationCoverageTracker()

        # -----------------------
        # Write manifest
        # -----------------------
        self._write_manifest()

    # =====================================================
    # Prescan
    # =====================================================
    def _prescan_dataset(self):
        valid = []

        for idx, fname in enumerate(self.image_files):
            path = os.path.join(self.images_dir, fname)
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    if img.width < self.min_size or img.height < self.min_size:
                        raise ValueError("Image too small")
                valid.append(idx)
                self.image_hashes[fname] = hash_file(path)
            except Exception:
                if self.strict:
                    raise

        self.valid_indices = valid

    # =====================================================
    # Split logic
    # =====================================================
    def _apply_split(self, split: str):
        setid_path = os.path.join(self.root_dir, "setid.mat")
        if not os.path.exists(setid_path):
            raise RuntimeError("setid.mat not found")

        sets = io.loadmat(setid_path)
        key_map = {"train": "trnid", "val": "valid", "test": "tstid"}

        key = key_map.get(split.lower())
        if key not in sets:
            raise RuntimeError(f"Split {split} not found in setid.mat")

        allowed = set((sets[key].flatten() - 1).tolist())

        new_valid = []
        for idx in self.valid_indices:
            fname = self.image_files[idx]

            if self.augmentation_version:
                src = self.provenance[fname]["source_file"]
                match = re.search(r"image_(\d+)", src)
            else:
                match = re.search(r"image_(\d+)", fname)

            if match and int(match.group(1)) - 1 in allowed:
                new_valid.append(idx)

        self.valid_indices = new_valid

    # =====================================================
    # Dataset API
    # =====================================================
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        fname = self.image_files[idx]
        path = os.path.join(self.images_dir, fname)

        if self.augmentation_version:
            src = self.provenance[fname]["source_file"]
            orig_idx = int(re.search(r"image_(\d+)", src).group(1)) - 1
        else:
            orig_idx = int(re.search(r"image_(\d+)", fname).group(1)) - 1

        label = int(self.labels[orig_idx])

        image = Image.open(path).convert("RGB")

        if self.use_augmentation and random.random() < 0.5:
            image = self.aug_transform(image)
        else:
            image = self.raw_transform(image)

        return image, label

    # =====================================================
    # Provenance
    # =====================================================
    def _load_provenance(self):
        prov_path = os.path.join(
            self.root_dir, "augmented", self.augmentation_version, "train", "provenance.csv"
        )
        with open(prov_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.provenance[row["augmented_file"]] = row

    # =====================================================
    # Manifest
    # =====================================================
    def _write_manifest(self):
        manifest = {
            "dataset": "OxfordFlowers102",
            "split": self.split,
            "offline_aug": bool(self.augmentation_version),
            "augmentation_version": self.augmentation_version,
            "num_samples": len(self.valid_indices),
            "created_at": datetime.utcnow().isoformat(),
            "dataset_hash": compute_dataset_hash(self.image_hashes),
        }

        name = f"dataset_{self.split or 'all'}_{self.augmentation_version or 'raw'}.json"
        with open(os.path.join(self.root_dir, name), "w") as f:
            json.dump(manifest, f, indent=2)


class MixedFlowersDataset(Dataset):
    """
    Train-only dataset that mixes raw + offline augmented samples.
    """

    def __init__(self, raw_dataset, aug_dataset, aug_ratio=0.6):
        assert raw_dataset.split == "train"
        assert aug_dataset.split == "train"

        self.raw = raw_dataset
        self.aug = aug_dataset
        self.aug_ratio = aug_ratio

    def __len__(self):
        return max(len(self.raw), len(self.aug))

    def __getitem__(self, idx):
        if random.random() < self.aug_ratio:
            return self.aug[idx % len(self.aug)]
        return self.raw[idx % len(self.raw)]
