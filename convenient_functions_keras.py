import numpy as np
from keras.layers import Input, concatenate, Conv2D, AveragePooling2D, Dropout, BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.keras.utils import Sequence
import cv2


def read_list_imgs(file_list, normalize=False):
    """
    Convenient function to read a list of images given in a file list
    """
    list_imgs = []
    for img_file in file_list:
        img_bgr = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        if normalize:
            img_bgr = img_bgr.astype("float32") / 255.

        list_imgs.append(img_bgr)

    return np.array(list_imgs)


class MyCustomGenerator(Sequence):
    """
    Convenient class to generate data for training GANNs models
    """
    def __init__(self, image_filenames, label_filenames, batch_size):
        # List of files corresponding inputs and labels as well as the size of the batch
        self.image_filenames = image_filenames
        self.label_filenames = label_filenames
        self.batch_size = batch_size

    def __len__(self) :
        # The number of files in the training batch
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx) :
        # Getting the number of files in the selected batch
        batch_x_list = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y_list = self.label_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # Read the images and the labels
        bgr_batch = read_list_imgs(batch_x_list, normalize=True)
        label_batch = read_list_imgs(batch_y_list, normalize=False)

        return bgr_batch, label_batch


def create_unet(insize=(320, 512), nclasses=4):
    # create unet network
    inputs = Input((insize[0], insize[1], 3))
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(inputs)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)  # 16

    conv2 = BatchNormalization(momentum=0.99)(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization(momentum=0.99)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(conv2)
    conv2 = Dropout(0.02)(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)  # 8

    conv3 = BatchNormalization(momentum=0.99)(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization(momentum=0.99)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(conv3)
    conv3 = Dropout(0.02)(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)  # 4

    conv4 = BatchNormalization(momentum=0.99)(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization(momentum=0.99)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(conv4)
    conv4 = Dropout(0.02)(conv4)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    conv5 = BatchNormalization(momentum=0.99)(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization(momentum=0.99)(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(conv5)
    conv5 = Dropout(0.02)(conv5)
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv4)

    pool4 = AveragePooling2D(pool_size=(2, 2))(pool3)  # 2
    pool5 = AveragePooling2D(pool_size=(2, 2))(pool4)  # 1

    conv6 = BatchNormalization(momentum=0.99)(pool5)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(conv6)

    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(conv6)
    up7 = (UpSampling2D(size=(2, 2))(conv7))  # 2
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(up7)
    merge7 = concatenate([pool4, conv7], axis=3)

    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(merge7)
    up8 = (UpSampling2D(size=(2, 2))(conv8))  # 4
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(up8)
    merge8 = concatenate([pool3, conv8], axis=3)

    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(merge8)
    up9 = (UpSampling2D(size=(2, 2))(conv9))  # 8
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same',
                kernel_initializer='he_normal')(up9)
    merge9 = concatenate([pool2, conv9], axis=3)

    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same',
                    kernel_initializer='he_normal')(merge9)
    up10 = (UpSampling2D(size=(2, 2))(conv10))  # 16
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same',
                    kernel_initializer='he_normal')(up10)

    conv11 = Conv2D(16, (3, 3), activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv10)
    up11 = (UpSampling2D(size=(2, 2))(conv11))  # 32
    conv11 = Conv2D(8, (3, 3), activation='relu', padding='same',
                    kernel_initializer='he_normal')(up11)

    conv12 = Conv2D(nclasses, (1, 1), activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv11)

    return inputs, conv12
