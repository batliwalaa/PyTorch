from collections import defaultdict
import numpy as np

class AugmentationCoverageTracker:
    def __init__(self):
        self.image_access = defaultdict(int)
        self.image_augmented = defaultdict(int)
        self.image_raw = defaultdict(int)
        self.image_time = defaultdict(int)

        self.class_access = defaultdict(int)
        self.class_augmented = defaultdict(int)

    def log_access(self, image_id, label, augmented, load_time):
        self.image_access[image_id] += 1
        self.image_time[image_id] += load_time

        if augmented:
            self.image_augmented[image_id] += 1
            self.class_augmented[label] += 1
        else:
            self.image_raw[image_id] += 1

        self.class_access[label] += 1
        
    def get_slow_images(self, min_samples=3, slow_factor=3.0):
        times = {
            img: self.image_time[img] / self.image_access[img]
            for img in self.image_access
            if self.image_access[img] >= min_samples
        }

        if not times:
            # Return a consistent tuple when there are no qualifying samples
            return {}, None

        median_time = np.median(list(times.values()))

        slow = {
            img: avg_time
            for img, avg_time in times.items()
            if avg_time > slow_factor * median_time
        }

        return slow, median_time
