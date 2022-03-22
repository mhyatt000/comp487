import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    MaxPool2D,
    GlobalAvgPool2D,
    Concatenate,
)
from tensorflow.keras.models import Sequential

# from d2l import tensorflow as d2l


def vgg_block(num_convs, num_channels):
    blk = Sequential()
    for _ in range(num_convs):
        blk.add(Conv2D(num_channels, kernel_size=3, padding="same", activation="relu"))
        blk.add(MaxPool2D(pool_size=2, strides=2))

    return blk


def vgg(conv_arch=()):

    if not conv_arch:
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    print(f"vgg-{3+sum([item[0] for item in conv_arch])}")

    net = Sequential()
    # The convulational part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    net.add(
        Sequential(
            [
                Flatten(),
                Dense(4096, activation="relu"),
                Dropout(0.5),
                Dense(4096, activation="relu"),
                Dropout(0.5),
                Dense(10),
            ]
        )
    )
    return net


def nin_block(num_channels, kernel_size, strides, padding):
    return Sequential(
        [
            Conv2D(
                num_channels,
                kernel_size,
                strides=strides,
                padding=padding,
                activation="relu",
            ),
            Conv2D(num_channels, kernel_size=1, activation="relu"),
            Conv2D(num_channels, kernel_size=1, activation="relu"),
        ]
    )


def nin():
    return Sequential(
        [
            nin_block(96, kernel_size=11, strides=4, padding="valid"),
            MaxPool2D(pool_size=3, strides=2),
            nin_block(256, kernel_size=5, strides=1, padding="same"),
            MaxPool2D(pool_size=3, strides=2),
            nin_block(384, kernel_size=3, strides=1, padding="same"),
            MaxPool2D(pool_size=3, strides=2),
            Dropout(0.5),
            # There are 10 label classes
            nin_block(10, kernel_size=3, strides=1, padding="same"),
            GlobalAveragePooling2D(),
            Reshape((1, 1, 10)),
            # Transform the four-dimensional output into two-dimensional output
            # with a shape of (batch size, 10)
            Flatten(),
        ]
    )


class Inception(tf.keras.Model):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4):
        super().__init__()

        self.p1_1 = Conv2D(c1, 1, activation="relu")

        self.p2_1 = Conv2D(c2[0], 1, activation="relu")
        self.p2_2 = Conv2D(c2[1], 3, padding="same", activation="relu")

        self.p3_1 = Conv2D(c3[0], 1, activation="relu")
        self.p3_2 = Conv2D(c3[1], 5, padding="same", activation="relu")

        self.p4_1 = MaxPool2D(3, 1, padding="same")
        self.p4_2 = Conv2D(c4, 1, activation="relu")

    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # Concatenate the outputs on the channel dimension
        return Concatenate()([p1, p2, p3, p4])


def googlenet():
    """GoogLeNet with inception blocks"""

    return Sequential([_b1(), _b2(), _b3(), _b4(), _b5(), Dense(10)])


def _b1():
    """GoogLeNet module 1"""

    return Sequential(
        [
            Conv2D(64, 7, strides=2, padding="same", activation="relu"),
            MaxPool2D(pool_size=3, strides=2, padding="same"),
        ]
    )


def _b2():
    """GoogLeNet module 2"""

    return Sequential(
        [
            Conv2D(64, 1, activation="relu"),
            Conv2D(192, 3, padding="same", activation="relu"),
            MaxPool2D(pool_size=3, strides=2, padding="same"),
        ]
    )


def _b3():
    """GoogLeNet module 3"""

    return Sequential(
        [
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            MaxPool2D(pool_size=3, strides=2, padding="same"),
        ]
    )


def _b4():
    """GoogLeNet module 4"""

    return Sequential(
        [
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            MaxPool2D(pool_size=3, strides=2, padding="same"),
        ]
    )


def _b5():
    """GoogLeNet module 5"""

    return Sequential(
        [
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            GlobalAvgPool2D(),
            Flatten(),
        ]
    )


def get_shapes(net):
    "prints output shapes to command line"

    X = tf.random.uniform((1, 28, 28, 1))
    for blk in net.layers:
        X = blk(X)
        print(blk.__class__.__name__, "output shape:\t", X.shape)


def main():
    """vgg-11"""
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    small_conv_arch = [(pair[0], pair[1] // 8) for pair in conv_arch]
    model = vgg(conv_arch)

    get_shapes(model)


if __name__ == "__main__":
    main()
