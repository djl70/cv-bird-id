import os
import shutil
import sys

#base_dir = "C:/Users/dlohr/Downloads/cv-bird-classification/CUB_200_2011"
base_dir = "./data"
train_dir = "./data/cropped/train"
test_dir = "./data/cropped/test"

# Create a dictionary of all files
files = {}
with open(os.path.join(base_dir, "images.txt")) as f:
    for line in f:
        (key, val) = line.split()
        files[int(key)] = val

# Get the recommended train/test splits
splits = {}
with open(os.path.join(base_dir, "train_test_split.txt")) as f:
    for line in f:
        (key, val) = line.split()
        splits[int(key)] = bool(int(val))

# Count the amount of data in each split
training_count = 0
testing_count = 0
for k,v in splits.items():
    if v:
        training_count += 1
    else:
        testing_count += 1

# Get a list of all classes (will be used to create a folder for each one)
classes = []
with open(os.path.join(base_dir, "classes.txt")) as f:
    for line in f:
        (_, c) = line.split()
        classes.append(c)

# Set up directory structure
if os.path.exists(train_dir) or os.path.exists(test_dir):
    sys.stderr.write("Training and/or testing directories already exist. Delete them and then run split_data.py again.")
    exit(-1)

os.mkdir(train_dir)
os.mkdir(test_dir)

for c in classes:
    os.mkdir(os.path.join(train_dir, c))
    os.mkdir(os.path.join(test_dir, c))

# Copy files to their respective folders
for k,v in files.items():
    path = os.path.join(train_dir, v) if splits[k] else os.path.join(test_dir, v)
    shutil.copy2(os.path.join(base_dir, "cropped", v), path)
