#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import cv2
import os
import pickle
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

def main(begin_path, model_folder):

    try:
        os.mkdir(model_folder)
    except:
        print 'OUTPUT FOLDER NOT CREATED'

    if not begin_path[len(begin_path)-1] == '/':
        begin_path += '/'

    classes = glob.glob(begin_path + '*')
    classes = sorted([f.split(begin_path)[1] for f in classes])
    end_path = '/*.png'

    datasetFilepaths = [begin_path + folder_name + end_path for folder_name in classes]

    imageSize = 28

    for datasetFilepath in datasetFilepaths:
        dataset = import_data(datasetFilepath, imageSize)
        try:
            x_train = np.append(x_train, dataset['x_train'], axis = 0)
            x_test = np.append(x_test, dataset['x_test'], axis = 0)
        except:
            x_train = dataset['x_train']
            x_test = dataset['x_test']

    input_img = Input(shape=(imageSize, imageSize, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    encoder = Model(input_img, encoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    earlyStopping=EarlyStopping(monitor='val_loss', patience=6000, verbose=0, mode='auto')

    history = autoencoder.fit(x_train, x_train,
                    epochs=10000,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose = 1,
                    callbacks=[earlyStopping])

    train_loss = history.history['loss']
    final_loss = train_loss[len(train_loss)-1]
    val_loss = history.history['val_loss']
    final_val_loss = val_loss[len(val_loss)-1]
    print ('iter: ' + str(iter) + ', loss: ' + '{0:.4}'.format(final_loss) + ', val_loss: ' + '{0:.4}'.format(final_val_loss) + ', epochs: ' + str(len(train_loss)))

    # save new network weights and configuration information under new name

    i = 1
    while os.path.exists(model_folder + '%sautoencoder.h5' % i):
        i += 1

    historyName = str(i) + 'historyDict'
    with open(model_folder + historyName, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    autoencoderName = str(i) + 'autoencoder.h5'
    encoderName = str(i) + 'encoder.h5'
    autoencoder.save(model_folder + autoencoderName)
    encoder.save(model_folder + encoderName)

    configName = str(i) + 'config.txt'
    with open(model_folder + configName, 'w') as configfile:
        for path in datasetFilepaths:
            configfile.write(path + '\n')
        configfile.write('\n')
        for layer in autoencoder.layers:
            configfile.write(layer.name + ' ' + str(layer.input_shape) +  ' ' + str(layer.output_shape) + '\n')

    print "Saved new autoencoder at " + model_folder + autoencoderName


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("datasetFolder", help="path to folder with image dataset. Should contain subfolders representing disjunct classes. If no class information (or regression dataset), one subfolder containing all images required.")
    parser.add_argument("modelFolder", help="path to folder to save trained cnn")

    args = parser.parse_args()

    begin_path = args.datasetFolder
    model_folder = args.modelFolder

    main(begin_path, model_folder)
