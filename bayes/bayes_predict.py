#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from math import log10
import glob
import argparse

def find_in_histogram(x, hist):
    n = sum(hist[0])
    epsilon = 0.01
    l0 = len(hist[0])
    l1 = len(hist[1])
    for i in range(l1-1):
        if (x >= hist[1][0+i] and x <= hist[1][1+i]):
            value = hist[0][i]
            if value == 0:
                value = epsilon
            return value/n
    return epsilon/n

def Bayes(P, pt1):
    sum_ = 0.0
    pB = {}
    for key in sorted(P.keys()):
        sum_ += P[key]*pt1[key]
    for key in sorted(P.keys()):
        p = float(P[key]*pt1[key])/sum_
        pB[key] = p
    return pB

def update_prior(input_, models, prior):
    log_p = dict((key, log10(prior[key])) for key in prior.keys())
    for model_key in sorted(models.keys()):
        for i in range(n_dim):
            update = find_in_histogram(input_[i], models[model_key][i])
            log_p[model_key] = log_p[model_key] + log10(update)
        log_p[model_key] /= (n_dim)
        P[model_key] = pow(10, log_p[model_key])
    p_bayes = Bayes(P, pt1)

def classifier(input_, models, prior, poster):
    P = update_prior(input_, models, prior)
    p_bayes = Bayes(P, pt1)

def main(modelNpz, filenames):
    overall = 0.0
    i_sets = 0.0

    N = {}
    with open(modelNpz, 'r+') as dataset:
        dataset.seek(0)
        npzfile = np.load(dataset)
        N = npzfile['arr_0'][0]

    for filename in filenames:
        p_dec = 0.9
        n_classes = len(N.keys())
        pt1 = dict((key, 1.0/n_classes) for key in N.keys())
        success = dict((key, 0.0) for key in N.keys())

        matrix = np.loadtxt(filename)
        values = matrix.T

        (n_sam, n_dim) = values.shape
        P = {}

        for input_ in values:
            log_p = dict((key, 0.0) for key in N.keys())
            for iModel, model_key in enumerate(sorted(N.keys())):
                for i in range(n_dim):
                    update = find_in_histogram(input_[i], N[model_key][i])
                    log_p[model_key] = log_p[model_key] + log10(update)
                log_p[model_key] /= (n_dim)
                P[model_key] = pow(10, log_p[model_key])
            p_bayes = Bayes(P, pt1)

            inverse = [(value, key) for key, value in p_bayes.items()]
            winner = max(inverse)[1]
            success[winner] += 1.0

        #print success.items()
        inverse = [(value, key) for key, value in success.items()]
        print max(inverse)[1], '{0:0.2f}'.format(max(inverse)[0]/sum(success.values()))
        overall += max(inverse)[0]/sum(success.values())
        i_sets += 1

    print '============================'
    print 'Average accuracy: {0:0.2f}'.format(overall / i_sets)
    print '============================'
    print '============================'


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("dataToPredict", help="path to input txt dataset, or folder where encoded image features are saved. Each class is saved into train and test .txt")
    parser.add_argument("modelNpz", help="path to trained bayes model")
    parser.add_argument("--test", help="only predict on test set", action="store_true")
    parser.add_argument("--train", help="only predict on train set", action="store_true")

    args = parser.parse_args()

    inputPath = args.dataToPredict
    modelNpz = args.modelNpz

    predict_only_test = args.test
    predict_only_train = args.train

    #print inputPath

    if '.txt' in inputPath:
        filenames = [inputPath]
    else:
        if not inputPath[len(inputPath)-1] == '/':
            inputPath += '/'
        filenames = sorted(glob.glob(inputPath + '*'))

    if predict_only_test:
        filenames = [f for f in filenames if 'test' in f]
    else:
        if predict_only_train:
            filenames = [f for f in filenames if 'train' in f]

    #print filenames

    main(modelNpz, filenames)
