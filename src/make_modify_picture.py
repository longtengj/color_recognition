import math
import os

import cv2
from PIL import Image


def make_picture(
        file,  # image matrix
):
    image = cv2.imread(file)
    h, w, n = image.shape
    _image = Image.open(file)
    _image = _image.convert('RGB')
    img_array = _image.load()
    print(w, h, n)
    print("RGB", img_array[w / 10, h / 2])
    # b = image[:, :, 0]
    # g = image[:, :, 1]
    # r = image[:, :, 2]
    # print(b, g, r)
    # B
    image[:, :, 0] = 53
    # G
    image[:, :, 1] = 99
    # R
    image[:, :, 2] = 33
    cv2.imwrite("./training_protein_dataset/2000p4/" + "2000p4.png", image)


# filename = './training_leukocytes_dataset/-/2000p4.png'
filename = './training_protein_dataset/2000p4/2000p4.png'

if __name__ == '__main__':
    make_picture(filename)
