#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def model(filename):
    N = []
    input_data = filename
    matrix = np.loadtxt(filename)
    for h in matrix:
        [n, bins, patches] = plt.hist(h, bins = 5)
        pom = [n, bins]
        N.append(pom)
    return N

def main(dataset_path, outputfile):
    if not '.npz' in outputfile:
        outputfile += '.npz'

    models_dict={}
    models_list=[]

    onlyfiles = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and "train" in f]

    for filename in sorted(onlyfiles):
        filepath = os.path.join(dataset_path, filename)
        model_out = model(filepath)
        classname = filename.replace('-train.txt','')
        models_dict[classname] = model_out

    models_list.append(models_dict)

    if not os.path.exists(outputfile):
        os.mknod(outputfile)
    with open(outputfile, 'r+') as dataset:
        np.savez(dataset, models_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("datasetFolder", help="path to folder with features dataset. Should contain train and test txt files representing disjunct classes.")
    parser.add_argument("outputModelNpz", help="path to trained bayes file")

    args = parser.parse_args()

    dataset_path = args.datasetFolder
    outputfile = args.outputModelNpz

    main(dataset_path, outputfile)
