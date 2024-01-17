from pathlib import Path
import numpy as np
import cv2

def one_hot_encoding(color_labels: np.ndarray) -> np.ndarray:
    labels = np.zeros((color_labels.shape[0], color_labels.shape[1]), dtype="uint8")
    labels[
        np.where(
            (color_labels[::, ::, 0] == 0) & 
            (color_labels[::, ::, 2] == 255) & 
            (color_labels[::, ::, 1] == 255)
            )
        ] = 1  # Yellow color_labels -> Sclera
    labels[
        np.where(
            (color_labels[::, ::, 0] == 0) & 
            (color_labels[::, ::, 1] == 255) & 
            (color_labels[::, ::, 2] == 0))
        ] = 2  # Green color_labels -> Iris
    labels[
        np.where(
            (color_labels[::, ::, 0] == 0) &
            (color_labels[::, ::, 1] == 0) & 
            (color_labels[::, ::, 2] == 255)
        )
            ] = 3  # Red in color_labels -> Pupil
    labels[np.where(labels == 0)] = 4 # The rest of the image is eyelids and clamps
    
    # Setting spurious segments in the rest of the image
    labels = cv2.medianBlur(labels, ksize=5)

    labels_one_hot_encoded = np.zeros(
        (color_labels.shape[0], color_labels.shape[1], 4)).astype("uint8")

    for ii in range(4):
        idx = np.where(labels == ii + 1)
        labels_one_hot_encoded[idx[0], idx[1], ii] = 1

    return labels_one_hot_encoded


list_rgb_img = list(Path(r"C:\Users\dukej\OneDrive - Cassini Technologies B.V\anchor_frame").glob(r"**\*.png"))
path_label_img = Path(r"C:\Users\dukej\OneDrive - Cassini Technologies B.V\anchor_frame_annotations")

path_bgr_out = Path(r"D:\gitProjects\segmentation_unet\data_set\images")
path_label_out = Path(r"D:\gitProjects\segmentation_unet\data_set\labels")
SAMPLE_FACT = 3

if __name__ == "__main__":
    for ii in list_rgb_img:
        print(f"Processing {ii.stem}")
        # Reading the images
        file_name = ii.name
        parent_folder = ii.parent.name
        bgr_img = cv2.imread(str(ii))
        bgr_label_img = cv2.imread(str(path_label_img.joinpath(parent_folder, file_name)))

        # One hot encoding the labels
        labels_one_hot_encoded = one_hot_encoding(bgr_label_img)

        # Anonymizing the output image
        list_str = ii.name.split(" ")
        name_out = f"{parent_folder}_{list_str[0][:2:]}_{list_str[1][:2:]}_{list_str[-1]}"
        out_img_path = str(path_bgr_out.joinpath(name_out))
        out_label_path = str(path_label_out.joinpath(name_out))

        # Preprocessing the images before storage
        img_bgr_out = bgr_img[::SAMPLE_FACT, ::SAMPLE_FACT]
        img_bgr_out = img_bgr_out[20:-20:, 64:-64:]
        img_labels_out = labels_one_hot_encoded[::SAMPLE_FACT, ::SAMPLE_FACT]
        img_labels_out = img_labels_out[20:-20:, 64:-64:]

        # Storing the images with the labels
        cv2.imwrite(out_img_path, img_bgr_out)
        cv2.imwrite(out_label_path, img_labels_out)
