from scipy import io
from collections import Counter
import math

def compute_class_balanced_copies(labels_mat_path, max_copies=8):
    labels = io.loadmat(labels_mat_path)["labels"][0] - 1
    class_counts = Counter(labels)

    max_count = max(class_counts.values())

    copies_per_class = {}
    for cls, count in class_counts.items():
        ratio = max_count / count
        copies = max(1, math.ceil(ratio))
        copies_per_class[cls] = min(copies, max_copies)

    return copies_per_class
