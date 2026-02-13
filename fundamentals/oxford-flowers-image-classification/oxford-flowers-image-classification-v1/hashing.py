import json
import hashlib
import os

def hash_augmentation_policy(policy_dict):
    """
    Stable hash for an augmentation policy definition.
    Order-independent, reproducible, version-safe.
    """
    canonical = json.dumps(policy_dict, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]

def compute_generation_hash(
    policy_hash: str,
    raw_dataset_hash: str,
    copies_per_class: dict,
    generator_version: str = "v1"
) -> str:
    """
    Docstring for compute_generation_hash
    
    :param policy_hash: Description
    :type policy_hash: str
    :param raw_dataset_hash: Description
    :type raw_dataset_hash: str
    :param copies_per_class: Description
    :type copies_per_class: dict
    :param generator_version: Description
    :type generator_version: str
    :return: Description
    :rtype: str
    """
    # Stable hash for an augmentation dataset generation run.
    # Order-independent, reproducible.
    # Reads the augmentation policy hash, raw dataset hash,
    # copies per class, and generator version.
    # Returns a SHA256 hex digest.

    payload = {
        "policy_hash": policy_hash,
        "raw_dataset_hash": raw_dataset_hash,
        "copies_per_class": canonicalize_copies_per_class(copies_per_class),
        "generator_version": generator_version
    }

    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def hash_raw_image_dir(images_dir: str) -> str:
    """
    Docstring for hash_raw_image_dir
    
    :param images_dir: Description
    :type images_dir: str
    :return: Description
    :rtype: str
    """
    # Stable hash for a directory of raw images.
    # Order-independent, reproducible.
    # Reads all .jpg files in the directory.
    # Returns a SHA256 hex digest.

    # Stream each file hash (avoid loading entire file into memory twice)
    h = hashlib.sha256()
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        path = os.path.join(images_dir, fname)
        try:
            file_hash_hex = hash_file(path)
            # Convert hex digest back to raw bytes for stable aggregation
            h.update(bytes.fromhex(file_hash_hex))
        except Exception:
            # Skip unreadable files but continue hashing others
            continue
    return h.hexdigest()

def hash_file(path, chunk_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_hash(image_hashes: dict):
    h = hashlib.sha256()
    for name in sorted(image_hashes.keys()):
        h.update(name.encode())
        h.update(image_hashes[name].encode())
    return h.hexdigest()


def canonicalize_copies_per_class(copies_per_class: dict) -> dict:
    return {
        int(k): int(v)
        for k, v in sorted(copies_per_class.items())
    }