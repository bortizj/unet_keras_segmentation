import numpy as np
import random
import cv2


class CostumeAffineTransform:
    def __init__(self, scales, angles, txs, tys):
        self.scales = scales
        self.angles = angles
        self.txs = txs
        self.tys = tys

    def __call__(self, img, labels):
        # Getting a random value for the transformation within the given range
        angle = self.angles[0] + (self.angles[1] - self.angles[0]) * random.random()
        scale = self.scales[0] + (self.scales[1] - self.scales[0]) * random.random()
        tx = self.txs[0] + (self.txs[1] - self.txs[0]) * random.random()
        ty = self.tys[0] + (self.tys[1] - self.tys[0]) * random.random()

        # Constructing the affine transformation matrix (rotation is around the center)
        center = (img.shape[1] / 2, img.shape[0] / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

        # Adding the translation component to the rotation matrix
        tr_mat = rot_mat + np.array(
            [[0, 0, scale * tx], [0, 0, scale * ty]], dtype="float32"
        )

        # Transforming the image and the labels image
        img_tr = cv2.warpAffine(
            img,
            tr_mat,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        labels_tr = cv2.warpAffine(
            labels,
            tr_mat,
            (labels.shape[1], labels.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return img_tr, labels_tr
