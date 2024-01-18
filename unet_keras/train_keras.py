from convenient_functions_keras import create_unet
from convenient_functions_keras import MyCustomGenerator
from keras.models import Model
from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2

import matplotlib
matplotlib.use(r"TKAgg")



# Verifying if Keras can run with the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the following GPU(s):")
    for gpu in gpus:
        print("  ", gpu.name)
else:
    print("TensorFlow is not using any GPUs.")


# Data path to the database
data_path = Path("D:\gitProjects\segmentation_unet\data_set")

# Necessary variables for the Unet training
checkpoint_filepath = str(data_path.joinpath("model", "checkpoint_BinaryCrossentropy"))
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    )
LEARNING_RATE = 0.0005
BATCH_SIZE = 20
EPOCHS = 100
IMG_SIZE = (320, 512)
N_CLASSES = 4
PERCENTAGE_TRAINING = 70 # 50 - 100

# Reading the images and the labels in the same order
rgb_filenames = list(data_path.joinpath("images").glob(r"*.png"))
label_filenames = []
for ii in rgb_filenames:
    filename = data_path.joinpath("labels", ii.name)
    label_filenames.append(filename)

# Getting random samples for the training and validation
np.random.seed(2023)
idx_training = np.random.choice(
    len(rgb_filenames), 
    int(PERCENTAGE_TRAINING * len(rgb_filenames) / 100), 
    replace=False
    ).tolist()
idx_validation = np.setdiff1d(
    np.arange(len(rgb_filenames)), idx_training
    ).tolist()

# Getting the file list from the random samples for training
rgb_training_list = list(map(rgb_filenames.__getitem__, idx_training))
labels_training_list = list(map(label_filenames.__getitem__, idx_training))

# Getting the file list from the random samples for testing
rgb_validation_list = list(map(rgb_filenames.__getitem__, idx_validation))
labels_validation_list = list(map(label_filenames.__getitem__, idx_validation))

# Creating the data generators for validation and testing
training_data_generator = MyCustomGenerator(rgb_training_list, labels_training_list, BATCH_SIZE)
validation_data_generator = MyCustomGenerator(rgb_validation_list, labels_validation_list, BATCH_SIZE)

# Creating the Unet model based on the image size and number of classes
inputs, conv12 = create_unet(insize=(IMG_SIZE[0], IMG_SIZE[1]), nclasses=N_CLASSES)
model = Model(inputs, conv12)
print(model.summary())

# Setting the optimizer settings
model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), 
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
    )
model.summary()

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tb_log")

# Training the Unet model using the data generators
history = model.fit(
    x=training_data_generator,
    validation_data=validation_data_generator,
    epochs=EPOCHS,
    shuffle=True,
    steps_per_epoch=int(len(rgb_training_list) // BATCH_SIZE),
    validation_steps=int(len(rgb_validation_list) // BATCH_SIZE),
    callbacks=[model_checkpoint_callback],
    )
