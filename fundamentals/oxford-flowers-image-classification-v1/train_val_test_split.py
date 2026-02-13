import os
import shutil
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

def train_val_test_split(
        dataset_root = "./oxford_102_flowers", 
        output_root = './data_split', 
        train_ratio = 0.70,
        val_ratio = 0.15,
        test_ratio = 0.15):

  images_dir = os.path.join(dataset_root, "jpg")
  labels_path = os.path.join(dataset_root, "imagelabels.mat")
  # Load labels
  labels = loadmat(labels_path)["labels"][0]  # shape (8189,)
  image_files = sorted(os.listdir(images_dir))

  # Sanity check
  assert len(labels) == len(image_files)

  # Create output structure
  for split in ["train", "val", "test"]:
      os.makedirs(os.path.join(output_root, split), exist_ok=True)

  # Group images by class
  class_to_images = {}

  for img_name, label in zip(image_files, labels):
      # labels are 1-indexed → convert to 0-index
      label = label - 1
      class_to_images.setdefault(label, []).append(img_name)
      
  # Split per class
  for class_id, images in class_to_images.items():

      train_imgs, temp_imgs = train_test_split(
          images,
          test_size=(1 - train_ratio),
          random_state=42
      )

      val_imgs, test_imgs = train_test_split(
          temp_imgs,
          test_size=test_ratio / (val_ratio + test_ratio),
          random_state=42
      )

      for split_name, split_imgs in [
          ("train", train_imgs),
          ("val", val_imgs),
          ("test", test_imgs),
      ]:
          class_dir = os.path.join(output_root, split_name, str(class_id))
          os.makedirs(class_dir, exist_ok=True)

          for img in split_imgs:
              src = os.path.join(images_dir, img)
              dst = os.path.join(class_dir, img)
              shutil.copy2(src, dst)

  print("✅ Dataset successfully split 70/15/15 with stratification.")
