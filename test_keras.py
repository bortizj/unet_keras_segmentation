# import matplotlib
# import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from convenient_functions_keras import MyCustomGenerator

import numpy as np
import cv2

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# matplotlib.use(r"TKAgg")

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
checkpoint_filepath = data_path.joinpath("model", "checkpoint_BinaryCrossentropy")

# Reading the images and the labels in the same order
rgb_filenames = list(data_path.joinpath("images").glob(r"*.png"))
label_filenames = []
for ii in rgb_filenames:
    filename = data_path.joinpath("labels", ii.name)
    label_filenames.append(filename)

BATCH_SIZE = 20
data_generator = MyCustomGenerator(rgb_filenames, label_filenames, BATCH_SIZE)

model = tf.keras.models.load_model(str(checkpoint_filepath))
pred = model.predict(data_generator)


# Plotting the data for comparison
for ii in range(pred.shape[0]):
    print(f"Storing image {rgb_filenames[ii].stem}")

    # Reading the reference images for displaying 
    input_data = cv2.imread(str(rgb_filenames[ii]), cv2.IMREAD_UNCHANGED)
    ground_truth = cv2.imread(str(label_filenames[ii]), cv2.IMREAD_UNCHANGED)

    # Starting the variables to store the top and row images
    out_top = input_data.copy()
    out_bottom = input_data.copy()

    # # Plotting the ground truth
    # count = 1
    # plt.subplot(2,5,count)
    # plt.imshow(input_data[::, ::, ::-1])
    # plt.xticks([])
    # plt.yticks([])
    # count += 1

    for jj in range(4):
        ground_truth_ = (255 * ground_truth[::, ::, jj]).astype("uint8")
        pseudo_color_img = cv2.applyColorMap(ground_truth_, cv2.COLORMAP_PARULA)
        out_top = np.hstack((out_top, pseudo_color_img))

        # plt.subplot(2,5,count)
        # plt.imshow(ground_truth[::, ::, jj], vmin=0, vmax=1)
        # plt.xticks([])
        # plt.yticks([])
        # count += 1 

    # # Plotting the predictions
    # plt.subplot(2,5,count)
    # plt.imshow(input_data[::, ::, ::-1])
    # plt.xticks([])
    # plt.yticks([])
    # count += 1

    for jj in range(4):
        pred_ = np.clip(255 * pred[ii, ::, ::, jj], 0, 255).astype("uint8")
        pseudo_color_img = cv2.applyColorMap(pred_, cv2.COLORMAP_PARULA)
        out_bottom = np.hstack((out_bottom, pseudo_color_img))

        # plt.subplot(2,5,count)
        # plt.imshow(pred[ii, ::, ::, jj], vmin=0, vmax=1)
        # plt.xticks([])
        # plt.yticks([])
        # count += 1

    # plt.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.01, wspace=0.01)
    # plt.show()

    out_img = np.vstack((out_top, out_bottom))
    out_path = data_path.joinpath("predictions_BinaryCrossentropy", rgb_filenames[ii].name)
    cv2.imwrite(str(out_path), out_img)
