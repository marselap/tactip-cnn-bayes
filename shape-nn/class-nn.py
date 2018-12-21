#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Activation
from keras.models import Model, load_model, Sequential
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import glob
import cv2
import random
import itertools
import os
import argparse

def import_data(dataset_path, set_train_test="train"):
    x = np.asarray([])
    y = np.asarray([])

    onlyfiles = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and set_train_test in f]

    for count, filename in enumerate(sorted(onlyfiles)):
        filepath = os.path.join(dataset_path, filename)
        class_data = np.loadtxt(filepath)
        class_data = class_data.T
        class_label = np.zeros(class_data.shape[0]) + count
        if x.size == 0:
            x = class_data
            y = class_label
        else:
            #print x
            x = np.concatenate((x, class_data))
            y = np.concatenate((y, class_label))

    return {"x":x, "y":y}

def binary_data(trainset, testset):
    lb = preprocessing.LabelBinarizer()
    lb.fit(np.concatenate((trainset, testset)))
    y_train = lb.transform(trainset)
    y_test = lb.transform(testset)
    return {"train":y_train, "test":y_test}

def main(dataset_path, outputfile):

    trainset = import_data(dataset_path, "train")
    testset = import_data(dataset_path, "test")
    x_train = trainset["x"]
    x_test = testset["x"]
    labeled_y = binary_data(trainset["y"], testset["y"])
    y_train = labeled_y["train"]
    y_test = labeled_y["test"]

    classes = y_train.shape[1]

    model = Sequential()
    model.add(Dense(16, input_dim=16, activation="relu"))
    model.add(Dense(64, activation="sigmoid"))
    model.add(Dense(32, activation="sigmoid"))
    model.add(Dense(classes))
    model.add(Activation(tf.nn.softmax))
    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=[metrics.categorical_accuracy])

    earlyStopping=EarlyStopping(monitor='val_loss', patience=500, verbose=0, mode='auto')

    model.fit(x_train, y_train, epochs=1500, batch_size = 64,validation_data=(x_test, y_test), callbacks=[earlyStopping])
    print model.evaluate(x_train, y_train)
    print model.evaluate(x_test, y_test)

    model.save(outputfile)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datasetFolder", help="path to folder with features dataset. Should contain train and test txt files representing disjunct classes.")
    parser.add_argument("outputfile", help="path to trained bayes file")

    args = parser.parse_args()

    dataset_path = args.datasetFolder
    outputfile = args.outputfile

    main(dataset_path, outputfile)
