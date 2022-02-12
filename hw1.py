"""
assignment 1
Matthew Hyatt
Feb 3 2021
"""

import time
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD


def get_dataset():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    train_x = train_x.reshape(-1, 28, 28, 1).astype("float32") / 255
    test_x = test_x.reshape(-1, 28, 28, 1).astype("float32") / 255

    return train_x, train_y, test_x, test_y


def build_model(activation):
    model = Sequential(
        [
            Flatten(input_shape=(28, 28)),
            Dense(100, activation=activation),
            Dense(100, activation=activation),
            Dense(10),
        ]
    )

    model.compile(
        optimizer=SGD(learning_rate=0.0001),
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

    model = build_model("relu")
    model.summary()

    timed_eval(*data, model=model)


def task2(data):

    for activation in ["relu", "sigmoid", "tanh"]:
        model = build_model(activation)
        print(activation)
        timed_eval(*data, model=model)


def main():

    data = [*get_dataset()]

    print("task1\n")
    task1(data)

    print("task2\n")
    task2(data)


if __name__ == "__main__":
    main()
