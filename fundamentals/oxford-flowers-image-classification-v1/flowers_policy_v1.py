FLOWER_TRAIN_AUGMENTATION_POLICY = {
    "RandomResizedCrop": {
        "size": 224,
        "scale": [0.7, 1.0],
        "ratio": [0.9, 1.1]
    },
    "RandomHorizontalFlip": {"p": 0.5},
    "RandomRotation": {"degrees": 20},
    "ColorJitter": {
        "brightness": 0.25,
        "contrast": 0.25,
        "saturation": 0.2,
        "hue": 0.03
    },
    "GaussianBlur": {
        "kernel_size": 3,
        "p": 0.2
    },
    "RandomErasing": {
        "p": 0.15,
        "scale": [0.02, 0.08],
        "ratio": [0.3, 3.3],
        "value": "random"
    },
    "Normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}
