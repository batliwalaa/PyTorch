import matplotlib.pyplot as plt
import torch
import numpy as np

def denormalize(img_tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
    """
    img_tensor: C x H x W tensor
    Returns: H x W x C numpy array suitable for matplotlib
    """
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
    img_np = img_np * np.array(std) + np.array(mean)   # Undo normalization
    img_np = np.clip(img_np, 0, 1)                     # Clip to [0,1] for display
    return img_np


def visualize_dataset_augmentations(dataset, 
                                    classes_to_show=5, 
                                    samples_per_class=4,
                                    mean=[0.485,0.456,0.406], 
                                    std=[0.229,0.224,0.225]):
    """
    Visualize augmentations across multiple classes.
    
    Args:
        dataset: OxfordFlowersDataset with augmentation transforms applied
        classes_to_show: Number of unique class labels to display
        samples_per_class: Number of augmented images per class
        mean, std: Normalization used in transforms (for denormalization)
    """
    # Map class label -> indices
    label_to_indices = {}
    for idx in range(len(dataset)):
      _, label = dataset[idx]
      if label not in label_to_indices:
        label_to_indices[label] = []
      if len(label_to_indices[label]) < samples_per_class:
        label_to_indices[label].append(idx)
      if len(label_to_indices) >= classes_to_show:
        break
    
    fig, axes = plt.subplots(classes_to_show, samples_per_class, figsize=(3*samples_per_class, 3*classes_to_show))

    for i, (label, indices) in enumerate(label_to_indices.items()):
      for j, idx in enumerate(indices):
        img, _ = dataset[idx]
        if isinstance(img, torch.Tensor):
          img_np = denormalize(img, mean=mean, std=std)
        else:
          img_np = np.array(img)/255.0
        
        ax = axes[i,j] if classes_to_show > 1 else axes[j]
        ax.imshow(img_np)
        ax.axis("off")
        if j == 0:
          ax.set_ylabel(f"Class {label}", fontsize=10)
    
    plt.tight_layout()
    plt.show()
