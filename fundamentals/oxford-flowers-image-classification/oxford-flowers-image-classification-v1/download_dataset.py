import os
from urllib import request
import tarfile

def download_dataset():
    """Download and extract the Oxford 102 Flowers dataset with labels."""

    image_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    split_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
    
    dataset_dir = "oxford_102_flowers"
    image_archive = os.path.join(dataset_dir, "102flowers.tgz")
    labels_file = os.path.join(dataset_dir, "imagelabels.mat")
    split_file = os.path.join(dataset_dir, "setid.mat")

    os.makedirs(dataset_dir, exist_ok=True)

    if not os.path.exists(image_archive):
        print("Downloading image dataset...")
        request.urlretrieve(image_url, image_archive)

    if not os.path.exists(labels_file):
        print("Downloading labels...")
        request.urlretrieve(labels_url, labels_file)

    if not os.path.exists(split_file):
        print("Downloading split information...")
        request.urlretrieve(split_url, split_file)

    images_dir = os.path.join(dataset_dir, "jpg")
    if not os.path.exists(images_dir):
        print("Extracting images...")
        with tarfile.open(image_archive, "r:gz") as tar:
            # Safely extract to avoid path traversal vulnerabilities
            def _is_within_directory(directory, target):
                abs_directory = os.path.realpath(directory)
                abs_target = os.path.realpath(target)
                return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

            for member in tar.getmembers():
                member_path = os.path.join(dataset_dir, member.name)
                if not _is_within_directory(dataset_dir, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
            tar.extractall(dataset_dir)

    print("Oxford 102 Flowers dataset is ready.")
    