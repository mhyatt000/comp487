import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import HeNormal, Ones, RandomNormal
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Conv2D, AveragePooling2D, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import L1, L1L2, L2
# import tensorflow_datasets as tfds

import nnets

'''

ideas

1x1 are full connections on the pixel
nxn are full connections on the pixel independent of others ... stride 0

'''


def get_dataset():
    '''TODO resize parameter? resize to 96 (original 224 but thats expensive)'''
    # ds = tfds.load('mnist', split='train', shuffle_files=True)

    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    train_x = train_x.reshape(-1, 28, 28, 1).astype("float32") / 255
    test_x = test_x.reshape(-1, 28, 28, 1).astype("float32") / 255

    return train_x, train_y, test_x, test_y


def compile_model(model):
    model.compile(
        optimizer="adam",
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy()],
    )
    return model


def main():

    train_x, train_y, test_x, test_y = get_dataset()

    models = []

    models.append(nnets.build_vgg())
    models.append(nnets.build_nin())
    models.append(nnets.build_googlenet())
    models.append(nnets.build_resnet())
    models.append(nnets.build_densenet())
    models.append(nnets.build_senet())

    for model in models:

        model = compile_model(model)
        model(tf.random.uniform((1, 28, 28, 1)))
        model.summary()

        start = time.perf_counter()

        '''TODO add early stopping'''
        hist = model.fit(train_x, train_y, batch_size=32, epochs=5, verbose=1)

        stop = time.perf_counter()
        print(f'Finished in {round(stop-start, 2)} seconds')

    '''TODO add save/load weights and/or save/load entire model'''

    '''TODO add layers:
        randombrightness
        randomcontrast
        randomcrop
        randomflip
        randomheight
        randomrotation
        randomtranslation
        randomwidth
        randomzoom
    '''


if __name__ == '__main__':
    main()
