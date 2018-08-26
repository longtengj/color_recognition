import cv2
import numpy as np


def training_data(data_source, img_name, data):
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
    with open(data, 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def get_color_data_source(img_name):
    data_source = ""
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
    return data_source


def color_histogram_of_training_image(img_name):
    # detect image color by using image file name to label training data
    data_source = get_color_data_source(img_name)

    training_data(data_source, img_name, 'training.data')


def get_urine_color_data_source(img_name):
    ds = ""
    if 'p1' in img_name:
        ds = 'p1'
    elif 'n1' in img_name:
        ds = 'n1'
    elif 'np' in img_name:
        ds = 'np'
    elif 'p1' in img_name:
        ds = 'p1'
    elif 'p2' in img_name:
        ds = 'p2'
    elif 'p3' in img_name:
        ds = 'p3'
    return ds


def get_urine_ph_color_data_source(img_name):
    data_source = ""
    if '50' in img_name:
        data_source = '50'
    elif '60' in img_name:
        data_source = '60'
    elif '65' in img_name:
        data_source = '65'
    elif '70' in img_name:
        data_source = '70'
    elif '75' in img_name:
        data_source = '75'
    elif '80' in img_name:
        data_source = '80'
    elif '85' in img_name:
        data_source = '85'
    return data_source


def color_histogram_of_training_leukocytes_image(img_name):
    # detect image color by using image file name to label training data
    data_source = get_urine_color_data_source(img_name)

    training_data(data_source, img_name, 'training_leukocytes.data')


def color_histogram_of_training_nitirite_image(img_name):
    # detect image color by using image file name to label training data
    data_source = get_urine_color_data_source(img_name)

    training_data(data_source, img_name, 'training_nitirite.data')


def color_histogram_of_training_ph_image(img_name):
    # detect image color by using image file name to label training data
    data_source = get_urine_ph_color_data_source(img_name)

    training_data(data_source, img_name, 'training_ph.data')


def get_urine_urobllinogen_color_data_source(img_name):
    data_source = ""
    if '2p1' in img_name:
        data_source = '2p1'
    elif '1n1' in img_name:
        data_source = '1n1'
    elif '02n1' in img_name:
        data_source = '02n1'
    elif '4p2' in img_name:
        data_source = '4p2'
    elif '8p3' in img_name:
        data_source = '8p3'
    return data_source


def color_histogram_of_training_urobllinogen_image(img_name):
    # detect image color by using image file name to label training data
    data_source = get_urine_urobllinogen_color_data_source(img_name)

    training_data(data_source, img_name, 'training_urobllinogen.data')


def get_urine_protein_color_data_source(img_name):
    data_source = ""
    if 'np' in img_name:
        data_source = 'np'
    elif 'n1' in img_name:
        data_source = 'n1'
    elif '30p1' in img_name:
        data_source = '30p1'
    elif '100p2' in img_name:
        data_source = '100p2'
    elif '300p3' in img_name:
        data_source = '300p3'
    elif '2000p4' in img_name:
        data_source = '2000p4'
    return data_source


def color_histogram_of_training_protein_image(img_name):
    # detect image color by using image file name to label training data
    data_source = get_urine_protein_color_data_source(img_name)

    training_data(data_source, img_name, 'training_protein.data')
