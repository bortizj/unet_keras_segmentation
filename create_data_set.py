from pathlib import Path
import numpy as np
import cv2

import tqdm


def one_hot_encoding(color_labels: np.ndarray) -> np.ndarray:
    """
    Convenient function to do one-hot encoding from own labelled data
    """
    # constructing an index labelled image from the pseudo color RGB
    labels = np.zeros((color_labels.shape[0], color_labels.shape[1]), dtype="uint8")
    # Yellow color_labels -> Sclera
    labels[
        np.where(
            (color_labels[::, ::, 0] == 0)
            & (color_labels[::, ::, 2] == 255)
            & (color_labels[::, ::, 1] == 255)
        )
    ] = 1
    # Green color_labels -> Iris
    labels[
        np.where(
            (color_labels[::, ::, 0] == 0)
            & (color_labels[::, ::, 1] == 255)
            & (color_labels[::, ::, 2] == 0)
        )
    ] = 2
    # Red in color_labels -> Pupil
    labels[
        np.where(
            (color_labels[::, ::, 0] == 0)
            & (color_labels[::, ::, 1] == 0)
            & (color_labels[::, ::, 2] == 255)
        )
    ] = 3
    # The rest of the image is eyelids and clamps
    labels[np.where(labels == 0)] = 4

    # Setting spurious segments in the rest of the image
    labels = cv2.medianBlur(labels, ksize=3)

    # Putting the encoded labels in the different channels
    labels_one_hot_encoded = np.zeros(
        (color_labels.shape[0], color_labels.shape[1], 4)
    ).astype("uint8")
    for ii in range(4):
        idx = np.where(labels == ii + 1)
        labels_one_hot_encoded[idx[0], idx[1], ii] = 255

    return labels_one_hot_encoded


# Path to folders
source_folder = Path(
    r"D:\gitProjects\segmentation_unet\data_set\__database_manual_segmentation"
)
data_folder = Path(r"D:\gitProjects\segmentation_unet\data_set\data")

# The expected sub-folders in the data folder
list_subfolders = ["0_predev", "1_dev", "2_posdev"]

if __name__ == "__main__":
    # Constructing the set of images for training and testing
    # Note that the images are down-sampled in here
    # We have noticed that the segmentation can be done in very low resolution images
    SAMPLING_FACTOR = 8

    # Images in 0_predev and 1_dev are used for training and 2_posdev for testing
    for ii in list_subfolders:
        list_source = list(source_folder.joinpath(ii, "input").glob("*.png"))

        # Storing the images
        if ii in ["0_predev", "1_dev"]:
            folder_store = data_folder.joinpath("training")
        else:
            folder_store = data_folder.joinpath("testing")

        folder_source = folder_store.joinpath("source")
        folder_source.mkdir(parents=True, exist_ok=True)

        folder_labels = folder_store.joinpath("labels")
        folder_labels.mkdir(parents=True, exist_ok=True)

        folder_composed = folder_store.joinpath("composed")
        folder_composed.mkdir(parents=True, exist_ok=True)

        # Reading the images and down-sampling them
        for jj in tqdm.tqdm(list_source):
            source_path = jj
            filename = jj.name
            labels_path = source_folder.joinpath(ii, "labels", filename)

            # Reading images
            source_img = cv2.imread(str(source_path))
            labels_img = cv2.imread(str(labels_path))
            one_hot_encoding_img = one_hot_encoding(labels_img)

            # Removing the interlacing and down-sampling the source image
            source_img = source_img[::2, ::2]
            source_img = cv2.GaussianBlur(source_img, ksize=(7, 7), sigmaX=5)
            source_img = source_img[::SAMPLING_FACTOR, ::SAMPLING_FACTOR]

            # down-sampling the One hot encoding image
            one_hot_encoding_img = one_hot_encoding_img[
                :: 2 * SAMPLING_FACTOR, :: 2 * SAMPLING_FACTOR
            ]

            # Composed image is just making sure that the one hot encoding image is correct
            top = np.hstack(
                [one_hot_encoding_img[::, ::, 0], one_hot_encoding_img[::, ::, 1]]
            )
            bot = np.hstack(
                [one_hot_encoding_img[::, ::, 2], one_hot_encoding_img[::, ::, 3]]
            )
            composed = np.vstack([top, bot])

            # Writing the images in disk
            cv2.imwrite(str(folder_source.joinpath(filename)), source_img)
            cv2.imwrite(str(folder_labels.joinpath(filename)), one_hot_encoding_img)
            cv2.imwrite(str(folder_composed.joinpath(filename)), composed)
