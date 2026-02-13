"""Production transforms for inference and evaluation.

Deterministic pipeline: Resize -> CenterCrop -> ToTensor -> Normalize.
Intended for validation, testing, and deployment where augmentations must be deterministic.
"""

from torchvision import transforms

transform_flowers_prod = transforms.Compose([
  transforms.Resize(256), 
  transforms.CenterCrop(224), 
  transforms.ToTensor(), 
  transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ) 
])
