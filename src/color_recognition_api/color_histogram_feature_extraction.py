#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

from PIL import Image
import os
import cv2
import numpy as np


# import matplotlib.pyplot as plt
# from scipy.stats import itemfreq
# from color_recognition_api import knn_classifier as knn_classifier


def color_histogram_of_test_image(test_src_image):
    # load the image
    image = test_src_image
    # 拆分成BGR三个通道
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1
        # 通道B的直方图
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
            print(feature_data)

    with open('test.data', 'w') as myfile:
        myfile.write(feature_data)


def color_histogram_of_training_image(img_name):
    # detect image color by using image file name to label training data
    if 'red' in img_name:
        data_source = 'red'
    elif 'yellow' in img_name:
        data_source = 'yellow'
    elif 'green' in img_name:
        data_source = 'green'
    elif 'orange' in img_name:
        data_source = 'orange'
    elif 'white' in img_name:
        data_source = 'white'
    elif 'black' in img_name:
        data_source = 'black'
    elif 'blue' in img_name:
        data_source = 'blue'
    elif 'violet' in img_name:
        data_source = 'violet'

    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():
    # red color training images
    for f in os.listdir('./training_dataset/red'):
        color_histogram_of_training_image('./training_dataset/red/' + f)

    # yellow color training images
    for f in os.listdir('./training_dataset/yellow'):
        color_histogram_of_training_image('./training_dataset/yellow/' + f)

    # green color training images
    for f in os.listdir('./training_dataset/green'):
        color_histogram_of_training_image('./training_dataset/green/' + f)

    # orange color training images
    for f in os.listdir('./training_dataset/orange'):
        color_histogram_of_training_image('./training_dataset/orange/' + f)

    # white color training images
    for f in os.listdir('./training_dataset/white'):
        color_histogram_of_training_image('./training_dataset/white/' + f)

    # black color training images
    for f in os.listdir('./training_dataset/black'):
        color_histogram_of_training_image('./training_dataset/black/' + f)

    # blue color training images
    for f in os.listdir('./training_dataset/blue'):
        color_histogram_of_training_image('./training_dataset/blue/' + f)

    # 白细胞 阴性 color training images
    for f in os.listdir('./training_leukocytes_dataset/-'):
        color_histogram_of_training_image('./training_leukocytes_dataset/-/' + f)
    # 白细胞 微量 color training images
    for f in os.listdir('./training_leukocytes_dataset/+-'):
        color_histogram_of_training_image('./training_leukocytes_dataset/+-/' + f)
    # 白细胞 少量 color training images
    for f in os.listdir('./training_leukocytes_dataset/+'):
        color_histogram_of_training_image('./training_leukocytes_dataset/+/' + f)
    # 白细胞 中量 color training images
    for f in os.listdir('./training_leukocytes_dataset/++'):
        color_histogram_of_training_image('./training_leukocytes_dataset/++/' + f)
    # 白细胞 大量 color training images
    for f in os.listdir('./training_leukocytes_dataset/+++'):
        color_histogram_of_training_image('./training_leukocytes_dataset/+++/' + f)

    # 亚硝酸盐 阴性 color training images
    for f in os.listdir('./training_nitirite_dataset/-'):
        color_histogram_of_training_image('./training_nitirite_dataset/-/' + f)
    # 亚硝酸盐 少量 color training images
    for f in os.listdir('./training_nitirite_dataset/+'):
        color_histogram_of_training_image('./training_nitirite_dataset/+/' + f)
    # 亚硝酸盐 中量 color training images
    for f in os.listdir('./training_nitirite_dataset/++'):
        color_histogram_of_training_image('./training_nitirite_dataset/++/' + f)

    # PH 50 color training images
    for f in os.listdir('./training_ph_dataset/50'):
        color_histogram_of_training_image('./training_ph_dataset/50/' + f)
    # PH 60 color training images
    for f in os.listdir('./training_ph_dataset/60'):
        color_histogram_of_training_image('./training_ph_dataset/60/' + f)
    # PH 70 color training images
    for f in os.listdir('./training_ph_dataset/70'):
        color_histogram_of_training_image('./training_ph_dataset/70/' + f)
    # PH 75 color training images
    for f in os.listdir('./training_ph_dataset/75'):
        color_histogram_of_training_image('./training_ph_dataset/75/' + f)
    # PH 80 color training images
    for f in os.listdir('./training_ph_dataset/80'):
        color_histogram_of_training_image('./training_ph_dataset/80/' + f)
    # PH 85 color training images
    for f in os.listdir('./training_ph_dataset/85'):
        color_histogram_of_training_image('./training_ph_dataset/85/' + f)

    # 尿胆原 0.2阴性 color training images
    for f in os.listdir('./training_urobllinogen_dataset/02-'):
        color_histogram_of_training_image('./training_urobllinogen_dataset/02-/' + f)
    # 尿胆原 1阴性 color training images
    for f in os.listdir('./training_urobllinogen_dataset/1-'):
        color_histogram_of_training_image('./training_urobllinogen_dataset/1-/' + f)
    # 尿胆原 2阳性 color training images
    for f in os.listdir('./training_urobllinogen_dataset/2+'):
        color_histogram_of_training_image('./training_urobllinogen_dataset/2+/' + f)
    # 尿胆原 4阳性 color training images
    for f in os.listdir('./training_urobllinogen_dataset/4++'):
        color_histogram_of_training_image('./training_urobllinogen_dataset/4++/' + f)
    # 尿胆原 8阳性 color training images
    for f in os.listdir('./training_urobllinogen_dataset/8+++'):
        color_histogram_of_training_image('./training_urobllinogen_dataset/8+++/' + f)

    # 蛋白质 阴性 color training images
    for f in os.listdir('./training_protein_dataset/-'):
        color_histogram_of_training_image('./training_protein_dataset/-/' + f)
    # 蛋白质 微量 color training images
    for f in os.listdir('./training_protein_dataset/+-'):
        color_histogram_of_training_image('./training_protein_dataset/+-/' + f)
    # 蛋白质 30阳性 color training images
    for f in os.listdir('./training_protein_dataset/30+'):
        color_histogram_of_training_image('./training_protein_dataset/30+/' + f)
    # 蛋白质 100阳性 color training images
    for f in os.listdir('./training_protein_dataset/100++'):
        color_histogram_of_training_image('./training_protein_dataset/1++/' + f)
    # 蛋白质 300阳性 color training images
    for f in os.listdir('./training_protein_dataset/300+++'):
        color_histogram_of_training_image('./training_protein_dataset/300+++/' + f)
    # 蛋白质 >2000阳性 color training images
    for f in os.listdir('./training_protein_dataset/2000++++'):
        color_histogram_of_training_image('./training_protein_dataset/2000++++/' + f)
