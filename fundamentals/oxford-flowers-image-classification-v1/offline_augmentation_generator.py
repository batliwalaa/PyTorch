import os
import csv
import json
import random
import numpy as np
import re
from typing import Dict
from collections import Counter
from scipy import io
from PIL import Image
from torchvision import transforms
from scipy.io import loadmat
from datetime import datetime
from hashing import hash_augmentation_policy, hash_raw_image_dir, compute_generation_hash

# ============================================================
# OFFLINE FLOWER AUGMENTATION POLICY (DATA GENERATION)
# ============================================================

FLOWER_OFFLINE_AUG_POLICY: Dict = {
    "geometry": {
        "RandomResizedCrop": {
            "size": 224,
            "scale": [0.7, 1.0],
            "ratio": [0.9, 1.1]
        },
        "RandomRotation": {
            "degrees": 20
        },
        "RandomHorizontalFlip": {
            "p": 0.5
        }
    },
    "appearance": {
        "ColorJitter": {
            "brightness": 0.25,
            "contrast": 0.25,
            "saturation": 0.2,
            "hue": 0.03
        },
        "GaussianBlur": {
            "kernel_size": 3,
            "p": 0.2
        }
    }
}

# ============================================================
# BUILD OFFLINE TRANSFORM (PIL ONLY)
# ============================================================

def build_offline_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=tuple(FLOWER_OFFLINE_AUG_POLICY["geometry"]["RandomResizedCrop"]["scale"]),
            ratio=tuple(FLOWER_OFFLINE_AUG_POLICY["geometry"]["RandomResizedCrop"]["ratio"])
        ),
        transforms.RandomHorizontalFlip(
            p=FLOWER_OFFLINE_AUG_POLICY["geometry"]["RandomHorizontalFlip"]["p"]
        ),
        transforms.RandomRotation(
            FLOWER_OFFLINE_AUG_POLICY["geometry"]["RandomRotation"]["degrees"]
        ),
        transforms.ColorJitter(
            **FLOWER_OFFLINE_AUG_POLICY["appearance"]["ColorJitter"]
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(
                kernel_size=FLOWER_OFFLINE_AUG_POLICY["appearance"]["GaussianBlur"]["kernel_size"]
            )],
            p=FLOWER_OFFLINE_AUG_POLICY["appearance"]["GaussianBlur"]["p"]
        )
    ])

def _offline_aug_dataset_exists(
    raw_images_dir: str,
    output_root: str,
    copies_per_class: dict,
    policy_hash: any
) -> bool:
    raw_dataset_hash = hash_raw_image_dir(raw_images_dir)
    generation_hash = compute_generation_hash(
        policy_hash=policy_hash,
        raw_dataset_hash=raw_dataset_hash,
        copies_per_class=copies_per_class,
        generator_version="v1"
    )

    version_name = f"v_{generation_hash}"
    out_dir = os.path.join(output_root, "augmented", version_name)
    dataset_manifest_path = os.path.join(out_dir, "dataset.json")

    if os.path.exists(dataset_manifest_path):
        print("Augmented dataset already exists")
        print(f"Version: {version_name}")
        print(f"Manifest: {dataset_manifest_path}")
        print("Skipping regeneration")
        return True
    
    print("Expected augmented dir:", out_dir)
    print("Manifest exists:", os.path.exists(dataset_manifest_path))

    return False
# ============================================================
# OFFLINE AUGMENTATION GENERATOR
# ============================================================

def generate_augmented_dataset(
    raw_images_dir: str,
    output_root: str,
    copies_per_class: dict,
):
    """
    Generates an offline augmented dataset with provenance tracking.

    Output structure:
    augmented/
      └── v_<hash>/
          ├── train/images/
          └── train/provenance.csv
    """

    POLICY_HASH = hash_augmentation_policy(FLOWER_OFFLINE_AUG_POLICY)
    setid = io.loadmat(os.path.join(output_root, "setid.mat"))

    train_ids = setid["trnid"].flatten() - 1  # 0-based
    train_ids = set(train_ids.tolist())

    # Compute raw dataset hash and generation hash up-front so we can
    # return the same hash whether we skip regeneration or create files.
    raw_dataset_hash = hash_raw_image_dir(raw_images_dir)
    generation_hash = compute_generation_hash(
        policy_hash=POLICY_HASH,
        raw_dataset_hash=raw_dataset_hash,
        copies_per_class=copies_per_class,
        generator_version="v1"
    )

    if _offline_aug_dataset_exists(
        raw_images_dir=raw_images_dir,
        output_root=output_root,
        copies_per_class=copies_per_class,
        policy_hash=POLICY_HASH
    ):
        print("Existing dataset is up-to-date with current policy and raw data. No regeneration needed.")
        return generation_hash  # return the hash for caller

    print("Starting offline augmentation generation...")

    version_name = f"v_{generation_hash}"
    out_dir = os.path.join(output_root, "augmented", version_name)
    images_out = os.path.join(out_dir, "train", "images")

    os.makedirs(images_out, exist_ok=True)
    write_policy_json(out_dir, FLOWER_OFFLINE_AUG_POLICY, POLICY_HASH)

    # Load labels
    labels_path = os.path.join(os.path.dirname(raw_images_dir), "imagelabels.mat")    
    if not os.path.exists(labels_path):
      raise FileNotFoundError(f"Missing labels file: {labels_path}")

    labels = loadmat(labels_path)["labels"][0] - 1
    transform = build_offline_transform()
    provenance_path = os.path.join(out_dir, "train", "provenance.csv")
    class_augmented_counts = Counter()

    with open(provenance_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "augmented_file",
                "source_file",
                "policy_hash",
                "seed",
                "policy",
                "class",
                "split"
            ]
        )
        writer.writeheader()

        for fname in sorted(os.listdir(raw_images_dir)):
            if not fname.lower().endswith(".jpg"):
                continue

            src_path = os.path.join(raw_images_dir, fname)

            try:
                img = Image.open(src_path).convert("RGB")
            except Exception:
                continue  # skip unreadable files safely

            match = re.search(r"image_(\d+)", fname)
            if not match:
                continue  # skip files that don't match expected pattern

            orig_idx = int(match.group(1)) - 1

            if orig_idx not in train_ids:
                continue  # skip val/test images

            label = labels[orig_idx]
            num_copies = copies_per_class.get(label, 1)

            for i in range(num_copies):
                seed = random.randint(0, 10**9)
                # Seed the RNG locally and restore global RNG state afterwards so
                # we don't pollute global randomness used elsewhere.
                _rand_state = random.getstate()
                random.seed(seed)
                try:
                    aug_img = transform(img)
                finally:
                    random.setstate(_rand_state)

                aug_name = f"{fname[:-4]}__aug{i}.jpg"
                aug_path = os.path.join(images_out, aug_name)

                aug_img.save(aug_path, quality=95, subsampling=0)

                writer.writerow({
                    "augmented_file": aug_name,
                    "source_file": fname,
                    "policy_hash": POLICY_HASH,
                    "seed": seed,
                    "policy": json.dumps(FLOWER_OFFLINE_AUG_POLICY),
                    "class": label,
                    "split": "train"
                })

                class_augmented_counts[label] += 1

    write_dataset_manifest(
        out_dir=out_dir,
        raw_images_dir=raw_images_dir,
        copies_per_class=copies_per_class,
        class_augmented_counts=class_augmented_counts,
        policy_hash=POLICY_HASH,
        raw_dataset_hash=raw_dataset_hash,
        generation_hash=generation_hash,
    )

    print(f"Offline augmentation complete")
    print(f"Version: {version_name}")
    print(f"Output: {images_out}")
    print(f"Provenance: {provenance_path}")

    return generation_hash

def write_policy_json(out_dir, policy_dict, policy_hash):
    policy_path = os.path.join(out_dir, "policy.json")

    payload = {
        "policy_name": "flower_augmentation",
        "policy_hash": policy_hash,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "framework": "torchvision",
        "transforms": policy_dict
    }

    with open(policy_path, "w") as f:
        json.dump(payload, f, indent=2)

def write_dataset_manifest(
    out_dir: str,
    raw_images_dir: str,
    copies_per_class: dict,
    class_augmented_counts: Counter,
    policy_hash: str,
    raw_dataset_hash: str,
    generation_hash: str,
    generator_version: str = "v1",
):
    # Recursively convert data to JSON-serializable Python types and make all dict keys strings.
    def make_json_safe(obj):
        # numpy scalar
        if isinstance(obj, np.generic):
            return obj.item()

        # dict -> ensure keys are strings
        if isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}

        # Counter or other mappings
        try:
            from collections.abc import Mapping
        except Exception:
            Mapping = dict

        if isinstance(obj, Mapping) and not isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}

        # list/tuple -> list
        if isinstance(obj, (list, tuple)):
            return [make_json_safe(v) for v in obj]

        # basic python types
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj

        # try to convert numbers
        try:
            return int(obj)
        except Exception:
            try:
                return float(obj)
            except Exception:
                return str(obj)

    copies_per_class_serializable = make_json_safe(copies_per_class)
    augmented_images_per_class_serializable = make_json_safe(dict(class_augmented_counts))

    manifest = {
        "dataset_name": "oxford_102_flowers_offline_augmented",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "version": os.path.basename(out_dir),
        "raw_images_dir": raw_images_dir,
        "num_raw_images": len([
            f for f in os.listdir(raw_images_dir)
            if f.lower().endswith(".jpg")
        ]),
        "num_augmented_images": sum(class_augmented_counts.values()),
        "label_source": "imagelabels.mat",
        "policy_hash": policy_hash,
        "policy_file": "policy.json",
        "provenance_file": "provenance.csv",
        "augmentation_strategy": "class_balanced_offline",
        "copies_per_class": copies_per_class_serializable,
        "augmented_images_per_class": augmented_images_per_class_serializable,
        "image_naming": "image_xxxxx__augN.jpg",
        "image_size": [224, 224],
        "raw_dataset_hash": raw_dataset_hash,
        "generation_hash": generation_hash,
        "generator_version": generator_version,
        "hash_algorithm": "sha256",
        "num_classes": len(class_augmented_counts),
        "splits_augmented": ["train"],
        "augmentation_scope": "train_only",
    }

    manifest_path = os.path.join(out_dir, "dataset.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset manifest written: {manifest_path}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

#if __name__ == "__main__":
#    RAW_DIR = "oxford_102_flowers/jpg"
#    OUTPUT_ROOT = "oxford_102_flowers"

#    generate_augmented_dataset(
#        raw_images_dir=RAW_DIR,
#        output_root=OUTPUT_ROOT,
#        copies_per_class={}
#    )
