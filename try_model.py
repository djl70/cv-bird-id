import os
import numpy as np
import keras
from keras.applications import vgg16, resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

# Load pre-trained models
vgg_model = vgg16.VGG16(weights="imagenet")  # include_top=False, input_shape=(image_size, image_size, 3))
# resnet_model = resnet50.ResNet50(weights="imagenet")

# Specify paths to data files
dir_data_base = "C:/Users/dlohr/Downloads/cv-bird-classification/CUB_200_2011"
dir_data_img = "images"
dir_data_seg = "segmentations"
dir_bird_file = "017.Cardinal/Cardinal_0014_17389"
path_to_bird_img = os.path.join(dir_data_base, dir_data_img, dir_bird_file + ".jpg")
path_to_bird_seg = os.path.join(dir_data_base, dir_data_seg, dir_bird_file + ".png")

# Load an image in PIL format
bird_original = load_img(path_to_bird_img, target_size=(224, 224))
plt.imshow(bird_original)
plt.show()

# Convert the PIL image to a numpy array
bird_numpy = img_to_array(bird_original)
plt.imshow(np.uint8(bird_numpy))
plt.show()

# Convert the image into batch format
bird_batch = np.expand_dims(bird_numpy, axis=0)
plt.imshow(np.uint8(bird_batch[0]))
plt.show()

# Prepare the image for the VGG model
bird_processed = vgg16.preprocess_input(bird_batch.copy())

# Get the predicted probabilities for each class
predictions = vgg_model.predict(bird_processed)
label = decode_predictions(predictions)
print(label)
