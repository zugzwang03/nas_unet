import keras_tuner as kt
import main
import unet
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

def build_model(hp):
    return unet.build_unet_model(hp)

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    hyperband_iterations=10,
    directory='my_dir',
    project_name='unet_nas'
)

train_data = main.train

# Extract images and masks from train_data
x_train = np.array([entry['image'] for entry in train_data])
y_train = np.array([entry['mask'] for entry in train_data])

# # Call tuner.search with the extracted data
# tuner.search(
#     x_train,
#     y_train,
#     epochs=10,
#     batch_size=32,
#     validation_split=0.2  # Or use a separate validation set
# )