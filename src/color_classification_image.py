#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 8th July 2018 - before Google inside look 2018 :)
# -------------------------------------------------------------------------
import imghdr

import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import skimage.io as io
from skimage import data_dir


def rotate(
        img  # image matrix
):
    # read the test image
    source_image = cv2.imread(img)
    prediction = 'n.a.'

    # checking whether the training data is ready
    PATH = './training.data'

    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print('training data is ready, classifier is loading...')
    else:
        print('training data is being created...')
        open('training.data', 'w')
        color_histogram_feature_extraction.training()
        print('training data is ready, classifier is loading...')

    # get the prediction
    color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
    prediction = knn_classifier.main('training.data', 'test.data')
    print('training data is ready, classifier is loading', prediction)


# cv2.putText(
#     source_image,
#     'Prediction: ' + prediction,
#     (15, 45),
#     cv2.FONT_HERSHEY_PLAIN,
#     3,
#     200,
# )
# print('training data is ready, classifier is loading', prediction)
# # Display the resulting frame
# cv2.imshow('color classifier', source_image)
# cv2.waitKey(0)

def cutimage(dir, suffix):
    for root, dirs, files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root, file)
            filesuffix = os.path.splitext(filepath)[1][1:]
            if filesuffix in suffix:  # 遍历找到指定后缀的文件名["jpg",png]等
                image = cv2.imread(file)  # opencv剪切图片
                rotate(image)


def eachFile(filepath):
    list = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        list.append(child)
    return list


def picResize():
    files = eachFile(filePath)
    for file in files:
        if imghdr.what(file) in ('bmp', 'jpg', 'png', 'jpeg'):  # 判断图片的格式
            # img = cv2.imread(file)  # 读取图片
            rotate(file)


if __name__ == '__main__':
    filePath = r".\segment\\"
    picResize()

