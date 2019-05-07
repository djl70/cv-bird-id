import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
import sys
import os


class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def load_bounding_boxes(file):
    bb = {}
    with open(file) as f:
        for line in f:
            (key, x, y, w, h) = line.split()
            bb[int(key)] = BoundingBox(round(float(x)), round(float(y)), round(float(w)), round(float(h)))
    return bb


def crop(img, bb):
    return img[bb.y:bb.y+bb.h, bb.x:bb.x+bb.w, :]


def crop_square_at_least(img, bb, min_sz):
    img_size = img.shape
    #if img_size[0] < 224 or img_size[1] < 224:
    #    sys.stderr.write("Error: image does not meet minimum size requirements")
    #    return None

    side_length = max(bb.w, bb.h, min_sz)
    side_length = min(side_length, img_size[0], img_size[1])

    expand_w = side_length - bb.w
    expand_h = side_length - bb.h

    x0 = bb.x - (expand_w // 2)
    y0 = bb.y - (expand_h // 2)

    x0_clamped = max(x0, 0)
    y0_clamped = max(y0, 0)

    x1 = x0_clamped + side_length
    y1 = y0_clamped + side_length

    x1_clamped = min(x1, img_size[1])
    y1_clamped = min(y1, img_size[0])

    return img[y0_clamped:y1_clamped, x0_clamped:x1_clamped, :]


base_dir = "C:/Users/dlohr/Downloads/cv-bird-classification/CUB_200_2011"
cropped_dir = "./data/cropped"

# Create a dictionary of all files and bounding boxes
files = {}
with open(os.path.join(base_dir, "images.txt")) as f:
    for line in f:
        (key, val) = line.split()
        files[int(key)] = val
bounding_boxes = load_bounding_boxes(os.path.join(base_dir, "bounding_boxes.txt"))

# Get a list of all classes (will be used to create a folder for each one)
classes = []
with open(os.path.join(base_dir, "classes.txt")) as f:
    for line in f:
        (_, c) = line.split()
        classes.append(c)

# Set up directory structure
if os.path.exists(cropped_dir):
    sys.stderr.write("Cropped directory already exists. Delete it and then run crop_to_bounding_box.py again.")
    exit(-1)

os.mkdir(cropped_dir)

for c in classes:
    os.mkdir(os.path.join(cropped_dir, c))

# Crop each image and save the result
for k,v in files.items():
    old_path = os.path.join(base_dir, "images", v)
    new_path = os.path.join(cropped_dir, v)
    old_img = load_img(old_path)
    old_img = img_to_array(old_img)/255
    new_img = crop(old_img, bounding_boxes[k])
    if new_img is None:
        sys.stderr.write("Error occurred at index " + str(k))
        exit(-1)
    plt.imsave(new_path, new_img)
