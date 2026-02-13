import os
import time
import torch
import sys
import types

# Avoid importing full matplotlib when running lightweight bench
if 'matplotlib' not in sys.modules:
    sys.modules['matplotlib'] = types.ModuleType('matplotlib')
    sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')

from torch.utils.data import DataLoader
from transform_train import transform_flowers_train, transform_flowers_train_offline_aug
from transform_prod import transform_flowers_prod
from oxford_flowers_dataset import OxfordFlowersDataset, MixedFlowersDataset


def main():
    ROOT = os.path.join(os.path.dirname(__file__), "oxford_102_flowers")

    cpu_workers = max(0, min(2, (os.cpu_count() or 1) - 1))
    pin_memory = False

    print(f"cpu_workers={cpu_workers}, pin_memory={pin_memory}")

    # Select training transform based on device: use offline (no random augment) on CPU
    selected_train_transform = (
        transform_flowers_train if torch.cuda.is_available() else transform_flowers_train_offline_aug
    )
    print("Selected train transform:", "online" if selected_train_transform is transform_flowers_train else "offline/deterministic")

    raw_ds = OxfordFlowersDataset(
        root_dir=ROOT,
        raw_transform=selected_train_transform,
        aug_transform=None,
        prescan=True,
        track_coverage=False,
    )

    aug_ds = OxfordFlowersDataset(
        root_dir=ROOT,
        raw_transform=transform_flowers_prod,
        aug_transform=None,
        prescan=False,
        track_coverage=False,
    )

    train_ds = MixedFlowersDataset(raw_ds, aug_ds, aug_ratio=0.6)

    # Use single-process data loading on Windows to avoid worker pickling errors
    loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print(f"Dataset length: {len(train_ds)}, loader num_workers={loader.num_workers}")

    t0 = time.time()
    batch = next(iter(loader))
    t1 = time.time()

    print(f"data-only load time: {t1-t0:.2f}s")


if __name__ == '__main__':
    main()
