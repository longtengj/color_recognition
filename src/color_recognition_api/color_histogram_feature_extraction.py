#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

import os

import cv2
import numpy as np

# import matplotlib.pyplot as plt
# from scipy.stats import itemfreq
# from color_recognition_api import knn_classifier as knn_classifier
from src.color_recognition_api.color_histogram_feature_extraction_urine import color_histogram_of_training_image, \
    color_histogram_of_training_leukocytes_image, color_histogram_of_training_nitirite_image, \
    color_histogram_of_training_protein_image, color_histogram_of_training_urobllinogen_image, \
    color_histogram_of_training_ph_image


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
    for f in os.listdir('./training_leukocytes_dataset/n1'):
        color_histogram_of_training_leukocytes_image('./training_leukocytes_dataset/n1/' + f)
    # 白细胞 微量 color training images
    for f in os.listdir('./training_leukocytes_dataset/np'):
        color_histogram_of_training_leukocytes_image('./training_leukocytes_dataset/np/' + f)
    # 白细胞 少量 color training images
    for f in os.listdir('./training_leukocytes_dataset/p1'):
        color_histogram_of_training_leukocytes_image('./training_leukocytes_dataset/p1/' + f)
    # 白细胞 中量 color training images
    for f in os.listdir('./training_leukocytes_dataset/p2'):
        color_histogram_of_training_leukocytes_image('./training_leukocytes_dataset/p2/' + f)
    # 白细胞 大量 color training images
    for f in os.listdir('./training_leukocytes_dataset/p3'):
        color_histogram_of_training_leukocytes_image('./training_leukocytes_dataset/p3/' + f)

    # 亚硝酸盐 阴性 color training images
    for f in os.listdir('./training_nitirite_dataset/n1'):
        color_histogram_of_training_nitirite_image('./training_nitirite_dataset/n1/' + f)
    # 亚硝酸盐 少量 color training images
    for f in os.listdir('./training_nitirite_dataset/p1'):
        color_histogram_of_training_nitirite_image('./training_nitirite_dataset/p1/' + f)
    # 亚硝酸盐 中量 color training images
    for f in os.listdir('./training_nitirite_dataset/p2'):
        color_histogram_of_training_nitirite_image('./training_nitirite_dataset/p2/' + f)

    # PH 50 color training images
    for f in os.listdir('./training_ph_dataset/50'):
        color_histogram_of_training_ph_image('./training_ph_dataset/50/' + f)
    # PH 60 color training images
    for f in os.listdir('./training_ph_dataset/60'):
        color_histogram_of_training_ph_image('./training_ph_dataset/60/' + f)
    # PH 65 color training images
    for f in os.listdir('./training_ph_dataset/65'):
        color_histogram_of_training_ph_image('./training_ph_dataset/65/' + f)
    # PH 70 color training images
    for f in os.listdir('./training_ph_dataset/70'):
        color_histogram_of_training_ph_image('./training_ph_dataset/70/' + f)
    # PH 75 color training images
    for f in os.listdir('./training_ph_dataset/75'):
        color_histogram_of_training_ph_image('./training_ph_dataset/75/' + f)
    # PH 80 color training images
    for f in os.listdir('./training_ph_dataset/80'):
        color_histogram_of_training_ph_image('./training_ph_dataset/80/' + f)
    # PH 85 color training images
    for f in os.listdir('./training_ph_dataset/85'):
        color_histogram_of_training_ph_image('./training_ph_dataset/85/' + f)

    # 尿胆原 0.2阴性 color training images
    for f in os.listdir('./training_urobllinogen_dataset/02n1'):
        color_histogram_of_training_urobllinogen_image('./training_urobllinogen_dataset/02n1/' + f)
    # 尿胆原 1阴性 color training images
    for f in os.listdir('./training_urobllinogen_dataset/1n1'):
        color_histogram_of_training_urobllinogen_image('./training_urobllinogen_dataset/1n1/' + f)
    # 尿胆原 2阳性 color training images
    for f in os.listdir('./training_urobllinogen_dataset/2p1'):
        color_histogram_of_training_urobllinogen_image('./training_urobllinogen_dataset/2p1/' + f)
    # 尿胆原 4阳性 color training images
    for f in os.listdir('./training_urobllinogen_dataset/4p2'):
        color_histogram_of_training_urobllinogen_image('./training_urobllinogen_dataset/4p2/' + f)
    # 尿胆原 8阳性 color training images
    for f in os.listdir('./training_urobllinogen_dataset/8p3'):
        color_histogram_of_training_urobllinogen_image('./training_urobllinogen_dataset/8p3/' + f)

    # 蛋白质 阴性 color training images
    for f in os.listdir('./training_protein_dataset/n1'):
        color_histogram_of_training_protein_image('./training_protein_dataset/n1/' + f)
    # 蛋白质 微量 color training images
    for f in os.listdir('./training_protein_dataset/np'):
        color_histogram_of_training_protein_image('./training_protein_dataset/np/' + f)
    # 蛋白质 30阳性 color training images
    for f in os.listdir('./training_protein_dataset/30p1'):
        color_histogram_of_training_protein_image('./training_protein_dataset/30p1/' + f)
    # 蛋白质 100阳性 color training images
    for f in os.listdir('./training_protein_dataset/100p2'):
        color_histogram_of_training_protein_image('./training_protein_dataset/100p2/' + f)
    # 蛋白质 300阳性 color training images
    for f in os.listdir('./training_protein_dataset/300p3'):
        color_histogram_of_training_protein_image('./training_protein_dataset/300p3/' + f)
    # 蛋白质 >2000阳性 color training images
    for f in os.listdir('./training_protein_dataset/2000p4'):
        color_histogram_of_training_protein_image('./training_protein_dataset/2000p4/' + f)
