"""
assignment 3
matt hyatt
feb 22
"""

import time
from pprint import pprint
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import HeNormal, Ones, RandomNormal
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    InputLayer,
    Conv2D,
    AveragePooling2D,
    Dropout,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import L1, L1L2, L2


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
    optimizer=SGD(),
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


def custom_lenet(*, filters, kernel_size):

    model = Sequential(
        [
            InputLayer(input_shape=(28, 28, 1)),
            Conv2D(filters=filters, kernel_size=kernel_size, activation="relu"),
            AveragePooling2D(),
            Conv2D(filters=filters, kernel_size=kernel_size, activation="relu"),
            AveragePooling2D(),
            Flatten(),
            Dropout(0.5),
            Dense(120, kernel_regularizer="L2", activation="relu"),
            Dropout(0.5),
            Dense(84, kernel_regularizer="L2", activation="relu"),
            Dense(10),
        ]
    )

    model.compile(
        optimizer=SGD(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy()],
    )

    return model


def timed_eval(train_x, train_y, test_x, test_y, *, model, verbose=1):

    start = time.time()

    hist = model.fit(train_x, train_y, batch_size=32, epochs=50, verbose=verbose)
    results = model.evaluate(test_x, test_y, batch_size=32, verbose=verbose)

    stop = time.time()

    data = {
        "train accuracy": round(hist.history["sparse_categorical_accuracy"][-1], 4),
        "test accuracy": round(results[1], 4),
        "time": int(stop - start),
    }

    pprint(data)
    return data


def task1(data):

    best = 0
    hparam = (0, 0)
    accs = []

    pairs = [(1, 1), (3, 3), (30, 3)]
    for (i, j) in pairs:
        print(i, j)
        model = custom_lenet(filters=i, kernel_size=j)
        # model.summary()
        #
        acc = timed_eval(*data, model=model, verbose=0)['test accuracy']

        hparam = (i, j) if acc > best else hparam
        best = acc if acc > best else best
        accs.append((i, j, acc))

    print("best: ", hparam, best)

    """
    (1 1) 0.8129
    (3 3) 0.9505
    (30 3) 0.9806
    """


def main():

    data = [*get_dataset()]

    task1(data)


if __name__ == "__main__":
    main()
