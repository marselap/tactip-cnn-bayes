# TacTip biomimetic percetion

The project implements CNN encoder for TacTip sensor output images. The encoder, obtained from a trained autoencoder, extracts 16 numerical features from the (preprocessed) sensor output image. 

Simple classifiers are implemented as benchmarks for encoder validation, applied to extracted features. The classifiers are used as perception algorithms. Naive Bayes Classifier built based on [1]. A simple multilayer perceptron trained on the same task.


## Getting Started

Clone the repo.

### Prerequisites

The following python modules are used in the project:

    - keras
    - sklearn
    - numpy, glob, cv2, os, pickle, argparse, random, itertools, math



## Running the tests

You can use a pre-trained CNN model ```5encoder.h5```. It should be placed in ```model-cnn```

Encoding the dataset:

```
python autoencoder/encode_images.py dataset-image/my-dataset-folder model-cnn/5\encoder.h5 dataset-feature/my-feature-folder
```

Training the Bayes classifier:

```
python bayes/train\_bayes\_models.py dataset-feature/my-feature-folder model-bayes/my-bayes-model.npz
```

Testing the Bayes classifier on the train set:

```
python bayes/bayes_predict.py dataset-feature/my-feature-folder/ model-bayes/my-bayes-model.npz --train
```

Testing the Bayes classifier on the test set:

```
python bayes/bayes_predict.py dataset-feature/my-feature-folder/ model-bayes/my-bayes-model.npz --test
```

## References

[1]: Lepora, Nathan F., Kirsty Aquilina, and Luke Cramphorn. "Exploratory Tactile Servoing With Active Touch." IEEE Robotics and Automation Letters 2.2 (2017): 1156-1163.
