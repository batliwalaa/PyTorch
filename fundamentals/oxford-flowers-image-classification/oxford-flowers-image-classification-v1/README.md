# Oxford 102 Flowers — Image Classification

This folder contains a small production-focused pipeline for training and evaluating image classifiers on the Oxford 102 Flowers dataset. The code includes data QA, optional offline augmentation generation with provenance, a dataset implementation that supports mixing raw and augmented images, training utilities in a notebook, and ONNX export/inference support.

> A reproducible, CPU-friendly image classification pipeline that goes beyond training accuracy and focuses on data quality, augmentation provenance, and deployment-ready inference.

## What problem this repo solves

- Provide a reproducible pipeline to train a classifier on the Oxford-102 dataset while addressing dataset imbalances and noisy images.
- Support both online (per-batch randomized) augmentation and offline (pre-generated) augmentation with provenance and stable hashing so experiments are reproducible.
- Provide dataset QA: prescan, bad-file reporting, eviction of slow/corrupted images, and augmentation coverage metrics.

## Why Oxford-102 is hard

- Fine-grained classes: 102 flower species with subtle visual differences.
- Class imbalance: some classes have many examples, others very few.
- Small dataset size: careful augmentation and validation are required to avoid overfitting.

## Key files

- `image-classification.ipynb` — end-to-end notebook for data prep, training, evaluation, and ONNX export.
- `offline_augmentation_generator.py` — create an offline augmented image corpus with provenance and manifest.
- `oxford_flowers_dataset.py` — `OxfordFlowersDataset` and `MixedFlowersDataset` with prescan, eviction, and coverage tracking.
- `transform_train.py`, `transform_prod.py` — augmentation and production transforms.
- `download_dataset.py` — download and safely extract the raw dataset.

## Open and run the notebook

`fundamentals/oxford-flowers-image-classification/oxford-flowers-image-classification-v1/image-classification.ipynb`. The notebook handles transform selection (GPU vs CPU), dataset construction, training loop, diagnostics, and ONNX export.

## Notes & recommendations

- Keep `raw_transform` and `aug_transform` separate: `raw_transform` must be deterministic for validation/test while `aug_transform` introduces randomness for training. `MixedFlowersDataset` composes raw + augmented datasets and provides a simple `aug_ratio` sampling mechanism.
- Use offline augmentation when running on CPU-only machines to reduce per-batch CPU overhead; use online augmentation (randomized transforms) on GPU-equipped machines for higher throughput.
- The code includes dataset QA (prescan, bad-file reports) and provenance for offline augmentation so experiments are auditable.

---

## Findings

- **Custom Residual CNN (trained from scratch)**
  - After 40 epochs:
    - Train Acc: 75.3%
    - Val Acc: 43.8%
    - Heavy overfitting
    - Slow convergence
    - Needed full 40 epochs to plateau

  - Observations:
    - Validation improves gradually with LR decay
    - Scheduler helps stabilize later epochs
    - But representation learning is weak
    - Gap ~30% between train and val → limited generalization

> Training from scratch on a small dataset (≈1k train images) is hard.

- **ResNet18 (ImageNet pretrained)**
  - After only 5 epochs:
  - Train Acc: 99.96%
  - Val Acc: 91.67%
  - Extremely fast convergence
  - Minimal overfitting
  - Half the batch time
- **Why the Difference Is So Large**
  - Pre-training ResNet - ResNet already knows
    - Edges
    - Textures
    - Shapes
    - Object Structure
  - Custom model starts from scratch

- **Datset size reality** - Uses setId.map to split train, val and test set - 1020 images in train set - 1020 images in val set - 6149 images in test set
  Training deep CNN's from scratch need millions ideally (50k, 100k can also be used for training)
  Custom model is underfed
