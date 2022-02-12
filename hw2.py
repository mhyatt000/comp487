"""
assignment 2
matt hyatt
feb 11
"""

import time
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.initializers import Ones, HeNormal, RandomNormal


def get_dataset():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    train_x = train_x.reshape(-1, 28, 28, 1).astype("float32") / 255
    test_x = test_x.reshape(-1, 28, 28, 1).astype("float32") / 255

    return train_x, train_y, test_x, test_y


def build_model(
    *,
    activation="relu",
    hidden=2,
    weights="glorot_uniform",
    regularizer=None,
    optimizer="adam",
):
    """conveniently builds model for different tests"""

    model = Sequential(
        [
            Flatten(input_shape=(28, 28)),
            *[
                Dense(
                    100,
                    activation=activation,
                    kernel_initializer=weights,
                    kernel_regularizer=regularizer,
                    name=f"hidden{i+1}",
                )
                for i in range(hidden)
            ],
            Dense(10),
        ]
    )

    model.compile(
        optimizer=optimizer,
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy()],
    )
    return model


def timed_eval(train_x, train_y, test_x, test_y, *, model):

    start = time.time()

    hist = model.fit(train_x, train_y, batch_size=32, epochs=50, verbose=0)
    results = model.evaluate(test_x, test_y, batch_size=32, verbose=0)

    stop = time.time()

    pprint(
        {
            "train accuracy": round(hist.history["sparse_categorical_accuracy"][-1], 4),
            "test accuracy": round(results[1], 4),
            "time": int(stop - start),
        }
    )


def task1(data):
    '''
    Task 1: Compare the performance of [2,3,4] layer MLPs on MNIST dataset

    Settings:
    Adam optimizer with 2e-4 learning rate.
    '''

    for i in [2, 3, 4]:
        model = build_model(hidden=i, optimizer=Adam(learning_rate=2e-4))
        print(f"\n{i} hidden layers")
        timed_eval(*data, model=model)


def task2(data):
    """
    Task 2: Compare the performance of 2-layer MLP when using different settings.

    Settings:
    Different weight initializations (at least two methods)
    Different regularizations (at least two methods)
    Different optimizers (at least two methods)
    """

    weights = [Ones(), HeNormal()]
    regularizer = [L1(), L2(), L1L2()]
    optimizers = [Adam(), RMSprop(), SGD()]

    settings = [[w, r, o] for w in weights for r in regularizer for o in optimizers]
    names = [[str(item).split(".")[-1].split(" ")[0] for item in s] for s in settings]

    '''
    best results: HeNormal, L1L2, Adam
        accuracy: 0.9794
    '''
    for i, (s, n) in enumerate(zip(settings, names)):
        print('\n', f'{i+1} of {len(settings)}...', n)
        model = build_model(weights=s[0], regularizer=s[1], optimizer=s[2])
        timed_eval(*data, model=model)


def main():

    data = [*get_dataset()]

    task1(data)
    task2(data)


if __name__ == "__main__":
    main()
