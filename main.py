import os
import numpy as np
import keras
from keras import models, layers, optimizers
from keras.applications import vgg16, resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

#mode = "train"
mode = "test"

# CUB_200_2011 dataset
#train_dir = "./data/cub-200-2011/train"
#val_dir = "./data/cub-200-2011/val"
#test_dir = "./data/cub-200-2011/test"
#classes_count = 200

# nabirds dataset
train_dir = "./data/nabirds/train"
val_dir = "./data/nabirds/val"
test_dir = "./data/nabirds/test"
classes_count = 555

# Load pre-trained models
image_size = 224

history = None

if mode == "train":
    # VGG16 base
    # vgg_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
    # base_model = vgg16.VGG16
    # trainable_layers = 4

    # ResNet50 base
    resnet_model = resnet50.ResNet50(weights="imagenet")
    base_model = resnet50.ResNet50
    trainable_layers = 10

    base_model = base_model(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))

    # Freeze all but the last 4 layers
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in base_model.layers:
        print(layer, layer.trainable)

    # Create our new model
    bird_model = models.Sequential()

    # Add the vgg convolutional base model
    bird_model.add(base_model)

    # Add new layers
    bird_model.add(layers.Flatten())
    bird_model.add(layers.Dense(1024, activation="relu"))
    bird_model.add(layers.Dropout(0.5))
    bird_model.add(layers.Dense(classes_count, activation="softmax"))

    # Show a summary of the model
    bird_model.summary()

    # Set up data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_batchsize = 100
    validation_batchsize = 10

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode="categorical"
    )

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=validation_batchsize,
        class_mode="categorical",
        shuffle=False
    )

    # Set up early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=0,
        mode="auto"
    )

    # Compile the model
    bird_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"]
    )

    # Train the model
    history = bird_model.fit_generator(
        train_generator,
        callbacks=[early_stop],
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_batchsize,
        verbose=1
    )

    # Save the model
    bird_model.save("bird_model_resnet50_224_cub-200-2011_last4.h5")

    exit(0)
elif mode == "test":
    bird_model = models.load_model("bird_model_vgg16_224_nabirds_last4.h5")
    bird_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"]
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    test_batchsize = 10

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=test_batchsize,
        class_mode="categorical"
    )

    history = bird_model.evaluate_generator(
        test_generator,
        steps=test_generator.samples / test_generator.batch_size,
        verbose=1
    )

    print(history)

    exit(0)


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
