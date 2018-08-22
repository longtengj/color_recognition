#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 8th July 2018 - before Google inside look 2018 :)
# -------------------------------------------------------------------------
import colorsys
import imghdr

import cv2
from PIL import Image
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path


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

def eachFile(filepath):
    list = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        list.append(child)
    return list


def get_dominant_color(image):
    max_score = 0.0001
    dominant_color = None
    for count, (r, g, b) in image.getcolors(image.size[0] * image.size[1]):
        # 转为HSV标准
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        y = (y - 16.0) / (235 - 16)

        # 忽略高亮色
        if y > 0.9:
            continue
        score = (saturation + 0.1) * count
        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)
    return dominant_color

# def cutPic():
#     image = image.convert('RGB')
#     image = Image.open(file)
#     img_array = image.load()
#     r, l, n = img.shape
#     print(r, l, n)
#     print(img_array[l / 10, r / 2])
#     print(img_array[l / 2, r / 2])
#     print(img_array[l * 9 / 10, r / 2])
#     img_ = cv2.imread(file)
#     originh, originw = img_.shape[:2]  # 获取图像的高和宽
#     gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)  # 转成灰度图像
#     ret, binary = cv2.threshold(gray, 100, 180, cv2.THRESH_BINARY)  # 将灰度图像转成二值图像
#
#     _, _contours, _hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
#     cv2.drawContours(img_, _contours, -1, (0, 0, 255), 3)
#     cv2.imshow("img", img_)
#     for i in range(0, len(_contours)):
#         x, y, w, h = cv2.boundingRect(_contours[i])
#         # cv2.rectangle(img, (x, y), (x + w, y + h), (153, 153, 0), 5)
#         # newimage = origin[y + 10:y + h - 5, x + 10:x + w - 5]  # 先用y确定高，再用x确定宽
#         newimage = img[y:y + h, x:x + w]  # 先用y确定高，再用x确定宽
#         nh, nw = newimage.shape[:2]  # 获取图像的高和宽
#         # if nw < originw:
#
#         resultRootdir = "./segment/cut/"
#         if not cv2.os.path.isdir(resultRootdir):
#             cv2.os.makedirs(resultRootdir)
#         cv2.imwrite(resultRootdir + str(i) + ".jpg", newimage)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()


def picResize():
    files = eachFile(filePath)
    for file in files:
        if imghdr.what(file) in ('bmp', 'jpg', 'png', 'jpeg'):  # 判断图片的格式
            img = cv2.imread(file)  # 读取图片
            print(file.title())
            rotate(file)

            # r, g, b = get_dominant_color(image)
            # val = hex(b + ((g << 8) & 0xff00) + ((r << 16) & 0xff0000))
            # print(val)
            # resultRootdir = "./segment/"
            # if not cv2.os.path.isdir(resultRootdir):
            #     cv2.os.makedirs(resultRootdir)
            # cv2.imwrite(resultRootdir + str(val) + ".jpg", img)


if __name__ == '__main__':
    filePath = r".\segment\\"
    picResize()
