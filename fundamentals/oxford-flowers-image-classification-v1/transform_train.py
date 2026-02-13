"""Training transforms and offline-friendly transform variants.

Includes an online augmentation pipeline with randomized ops and an
offline/deterministic variant suitable for CPU-only benchmarking or
when per-batch randomness should be avoided.
"""

from torchvision import transforms

transform_flowers_train_offline_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


transform_flowers_train = transforms.Compose([
  transforms.RandomResizedCrop( 
    224, 
    scale=(0.7, 1.0), 
    ratio=(0.9, 1.1) 
  ), 
  transforms.RandomHorizontalFlip(p=0.5), 
  transforms.RandomRotation(20), 
  transforms.ColorJitter( 
    brightness=0.25, 
    contrast=0.25, 
    saturation=0.2, 
    hue=0.03 # VERY small on purpose 
  ), 
  transforms.RandomApply( 
    [transforms.GaussianBlur(kernel_size=3)], 
    p=0.2 
  ),
  transforms.ToTensor(),
  transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
  transforms.RandomErasing( 
    p=0.15, 
    scale=(0.02, 0.08), 
    ratio=(0.3, 3.3), 
    value='random' 
  ), 
  transforms.Normalize( mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225] ) 
])
