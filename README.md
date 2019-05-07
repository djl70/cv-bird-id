# Automatic Identification of Bird Species using Computer Vision

## Usage

The following steps should mostly work, but there may be additional steps you need to take to make things work perfectly. This certainly is not the cleanest code you'll see. I learned Python and Keras as I went.

### 1. Install Keras on Windows 10

1. Install [CUDA 10.0](https://developer.nvidia.com/compute/cuda/10.0/Prod/network_installers/cuda_10.0.130_win10_network)
2. Sign up for the [NVIDIA Developer Program](https://developer.nvidia.com/nvidia-developer-zone)
3. Download [cuDNN](https://developer.nvidia.com/cudnn) (v7.5.1 for CUDA 10.0)
4. Follow the instructions [here](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows) to install cuDNN
5. Finally, install [TensorFlow](https://www.tensorflow.org/install/gpu) and Keras:
```bash
pip3 install tensorflow-gpu
pip3 install keras
```

### 2. Download the datasets

- [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [NABirds](http://dl.allaboutbirds.org/nabirds)

### 3. Split the data

1. Run split_data_cub.py and/or split_data_nabirds.py to split the CUB-200-2011 and NABirds datasets, respectively (some modifications may need to be made to the paths)
2. An extra directory named images is made in this process. I moved the child folders of that directory to the same level and then deleted it

### 4. Train the model

1. Make sure `mode = "train"` is set inside main.py
2. Uncomment the lines for the dataset you wish to use
3. Uncomment the lines for the base model you wish to use
4. Rename the file the model will be saved to
5. Run main.py

### 5. Test the model

1. Make sure `mode = "test"` is set inside main.py
2. Uncomment the lines for the dataset you wish to use
3. Make sure the name of the model being loaded matches the one you saved after training
4. Run main.py. The output is the loss and accuracy

## Unused/outdated files

I kept these files here as a remnant of other things I tried. They are not used.
- crop_to_bounding_box.py
- nabirds.py
- split_data.py
- try_model.py
