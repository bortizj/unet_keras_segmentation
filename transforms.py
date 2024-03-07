import numpy as np
import random
import cv2


def _get_rand_value(in_list, default=0):
    # returns the random number between the range or list of values
    if in_list is None:
        answer = default
    elif len(in_list) == 2:
        answer = in_list[0] + (in_list[1] - in_list[0]) * random.random()
    else:
        answer = random.choice(in_list)

    return answer


class CostumeAffineTransform:
    def __init__(self, scales, angles, txs, tys):
        self.scales = scales
        self.angles = angles
        self.txs = txs
        self.tys = tys

    def __call__(self, img, labels):
        # Getting a random value for the transformation within the given range or list
        scale = _get_rand_value(self.scales, default=1)
        angle = _get_rand_value(self.angles, default=0)
        tx = _get_rand_value(self.txs, default=0)
        ty = _get_rand_value(self.tys, default=0)

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
