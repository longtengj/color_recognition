#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 8th July 2018 - before Google inside look 2018 :)
# -------------------------------------------------------------------------
import colorsys
import imghdr
import os
import os.path

import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier


def do_classify(
        img,  # image matrix
        train_data_file_name  # image matrix
):
    # read the test image
    source_image = cv2.imread(img)
    prediction = 'n.a.'

    # checking whether the training data is ready
    PATH = './training.data'
    PATH = train_data_file_name

    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        # print('training data is ready, classifier is loading...')
        print("")
    else:
        print('training data is being created...')
        open('training.data', 'w')
        color_histogram_feature_extraction.training()
        print('training data is ready, classifier is loading...')

    # get the prediction
    color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
    prediction = knn_classifier.main(train_data_file_name, 'test.data')
    print('training data is over,result', train_data_file_name, prediction)

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


def each_file(filepath):
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


def foreach_picture(train_datas):
    files = each_file(filePath)
    for file in files:
        if imghdr.what(file) in 'png':  # 判断图片的格式

            # img = cv2.imread(file)  # 读取图片
            # print(file.title())
            file_name = os.path.split(file)[-1].split('.')[0]

            # for x in range(len(training_feature_vector)):
            index = int(file_name)
            print(index)
            if index < len(train_datas):
                do_classify(file, train_datas[index])

            # r, g, b = get_dominant_color(image)
            # val = hex(b + ((g << 8) & 0xff00) + ((r << 16) & 0xff0000))
            # print(val)
            # resultRootdir = "./segment/"
            # if not cv2.os.path.isdir(resultRootdir):
            #     cv2.os.makedirs(resultRootdir)
            # cv2.imwrite(resultRootdir + str(val) + ".jpg", img)


train_datas = ['training_leukocytes.data', 'training_nitirite.data',
               'training_urobllinogen.data', 'training_protein.data',
               'training_ph.data', 'training_leukocytes.data', 'training_nitirite.data',
               'training_urobllinogen.data', 'training_protein.data',
               'training_ph.data', 'training_ph.data']

if __name__ == '__main__':
    filePath = r".\segment\\"

    foreach_picture(train_datas)
