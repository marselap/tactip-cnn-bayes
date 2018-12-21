#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import cv2
import random
import itertools
import os
import argparse

def import_data(folderPath, resize_size):
    imagePaths = glob.glob(folderPath)
    data = np.empty((len(imagePaths), resize_size, resize_size), dtype=np.float32)
    for i, impath in enumerate(imagePaths):
        imgLarge = cv2.imread(impath)
        imgBW = cv2.cvtColor(imgLarge, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(imgBW, (resize_size, resize_size))
        data[i, ...] = img
    data = data.astype('float32') / 255.
    data = np.round(data + 0.3)
    data.resize(len(imagePaths), resize_size, resize_size, 1)
    seed = 20
    x_train, x_test = train_test_split(data, test_size=0.4, random_state=seed)
    return {'x_train':x_train, 'x_test':x_test}

def main(encoderName, begin_path, histdataFolder):

    if not histdataFolder[len(histdataFolder)-1] == '/':
        histdataFolder += '/'
    try:
        os.mkdir(histdataFolder)
    except:
        print 'OUTPUT FOLDER NOT CREATED'

    encoder = load_model(encoderName)

    if not begin_path[len(begin_path)-1] == '/':
        begin_path += '/'
    classes = glob.glob(begin_path + '*')
    classes = sorted([f.split(begin_path)[1] for f in classes])
    end_path = '/*.png'

    datasetFilepaths = [begin_path + folder_name + end_path for folder_name in classes]

    imageSize = 28

    for class_, datasetFilepath in zip(classes, datasetFilepaths):
        dataset = import_data(datasetFilepath, imageSize)
        hist_train = dataset['x_train']
        hist_test = dataset['x_test']

        histdata = [hist_train, hist_test]
        histpaths = [class_ + '-train.txt', class_ + '-test.txt']
        for (dataset, filepath) in zip(histdata, histpaths):
            data = []
            for img in dataset:
                img.resize(1, 28, 28, 1)
                img_enc = encoder.predict(img)
                dmy = list(itertools.chain.from_iterable(img_enc))
                dmy = list(itertools.chain.from_iterable(dmy))
                img_enc_1d = list(itertools.chain.from_iterable(dmy))
                data.append(img_enc_1d)

            tmp = map(list, zip(*data)) #transpose list of lists
            histdatafile = histdataFolder + filepath

            with open(histdatafile,'a') as datafile:
                for var in tmp:
                    for item in var:
                        datafile.write('%s ' % item)
                    datafile.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("datasetFolder", help="path to folder with images dataset. Should contain subfolders representing disjunct classes. If no class information (or regression dataset), one subfolder containing all images required. Subfolder names taken as class labels.")
    parser.add_argument("encoderModel", help="path to trained encoder cnn")
    parser.add_argument("featuresOutputFolder", help="path to folder where encoded image features are saved. Each class is saved into train and test .txt")

    args = parser.parse_args()

    begin_path = args.datasetFolder
    enc_model = args.encoderModel
    histdataFolder = args.featuresOutputFolder

    main(enc_model, begin_path, histdataFolder)
