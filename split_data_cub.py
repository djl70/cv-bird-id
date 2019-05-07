import os
import shutil
import sys
import random
import errno

def load_class_names(dataset_path):
    classes = {}

    with open(os.path.join(dataset_path, "classes.txt")) as f:
        for line in f:
            (k, c) = line.split()
            classes[int(k)] = c

    return classes


def load_image_labels(dataset_path):
    labels = {}

    with open(os.path.join(dataset_path, "image_class_labels.txt")) as f:
        for line in f:
            (k, c) = line.split()
            labels[int(k)] = int(c)

    return labels


def load_image_paths(dataset_path, path_prefix=''):
    paths = {}

    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            (k, p) = line.split()
            path = os.path.join(path_prefix, p)
            paths[int(k)] = path

    return paths


def split_each_class(class_names, image_labels, split_train=0.60, split_val=0.20, split_test=0.20):
    splits = {}

    for c in class_names.keys():
        # Find all images with label c
        class_images = [k for k,v in image_labels.items() if v == c]

        # Count images with label c
        class_count = len(class_images)

        # Split 60/20/20 train/val/test
        train_count = round(class_count * split_train)
        val_count = round(class_count * split_val)
        test_count = round(class_count * split_test)

        image_indices = list(range(class_count))
        random.shuffle(image_indices)

        train_indices = image_indices[0:train_count]
        val_indices = image_indices[train_count:train_count+val_count]
        test_indices = image_indices[train_count+val_count:]

        for i in train_indices:
            splits[class_images[i]] = 0
        for i in val_indices:
            splits[class_images[i]] = 1
        for i in test_indices:
            splits[class_images[i]] = 2

    return splits


def copy_by_split(class_splits, image_paths, source_base, destination_base):
    folders = {0: "train", 1: "val", 2: "test"}

    for k,v in class_splits.items():
        old_path = os.path.join(source_base, image_paths[k])
        new_path = os.path.join(destination_base, folders[v], image_paths[k])
        try:
            shutil.copy2(old_path, new_path)
        except IOError as e:
            if e.errno != errno.ENOENT:
                raise
            os.makedirs(os.path.dirname(new_path))
            shutil.copy2(old_path, new_path)


dataset_path = "C:/Users/dlohr/Downloads/cv-bird-classification/CUB_200_2011"
image_path_prefix = "images"
destination_path = "./data/cub-200-2011"

class_names = load_class_names(dataset_path)
image_labels = load_image_labels(dataset_path)
image_paths = load_image_paths(dataset_path, image_path_prefix)

class_splits = split_each_class(class_names, image_labels, 0.60, 0.20, 0.20)
copy_by_split(class_splits, image_paths, dataset_path, destination_path)
