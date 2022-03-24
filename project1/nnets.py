import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    MaxPool2D,
    AveragePooling2D,
    GlobalAvgPool2D,
    Concatenate,
    BatchNormalization,
    Activation,
    ReLU,
    Reshape
)
from tensorflow.keras.models import Sequential
import numpy as np

"""VISUAL GEOMETRY GROUP"""


def vgg_block(num_convs, num_channels):
    blk = Sequential()
    for _ in range(num_convs):
        blk.add(Conv2D(num_channels, kernel_size=3, padding="same", activation="relu"))
        blk.add(MaxPool2D(pool_size=2, strides=2))

    return blk


def build_vgg(conv_arch=()):

    if not conv_arch:
        conv_arch = ((1, 64), (1, 128), (1, 256))
        # # too many for mnist
        # conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
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


"""NETWORK IN NETWORK"""


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


def build_nin():
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
            GlobalAvgPool2D(),
            Reshape((1, 1, 10)),
            # Transform the four-dimensional output into two-dimensional output
            # with a shape of (batch size, 10)
            Flatten(),
        ]
    )


"""INCEPTION"""


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


def build_googlenet():
    """GoogLeNet with inception blocks"""

    return Sequential([_b1(), _b2(), _b3(), _b4(), _b5(), Dense(10)])


def _b1():
    """GoogLeNet module 1"""

    return Sequential(
        [
            Conv2D(64, 7, strides=2, padding="same"),
            BatchNormalization(),
            ReLU(),
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


"""RESNET"""


class Residual(tf.keras.Model):
    """The Residual block of ResNet."""

    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, padding="same", kernel_size=3, strides=strides
        )
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding="same")
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides
            )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(
        self, num_channels, num_residuals, first_block=False, use_se=False, **kwargs
    ):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2)
                    if not use_se
                    else SEResidual(num_channels, use_1x1conv=True, strides=2)
                )
            else:
                self.residual_layers.append(Residual(num_channels)
                                            if not use_se else SEResidual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X


def build_resnet(use_se=False):

    return Sequential(
        [
            _b1(),
            ResnetBlock(64, 2, first_block=True, use_se=use_se),
            ResnetBlock(128, 2, use_se=use_se),
            ResnetBlock(256, 2, use_se=use_se),
            ResnetBlock(512, 2, use_se=use_se),
            GlobalAvgPool2D(),
            Dense(10),
        ]
    )


"""DENSENET"""


class ConvBlock(tf.keras.layers.Layer):
    """like resudual block but it appends layers instead of adding"""

    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.conv = Conv2D(filters=num_channels, kernel_size=(3, 3), padding="same")

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y


class DenseBlock(tf.keras.layers.Layer):
    """implements ConvBlock"""

    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x


class TransitionBlock(tf.keras.layers.Layer):
    """manages exploding growth rate from DenseBlock"""

    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = BatchNormalization()
        self.relu = ReLU()
        self.conv = Conv2D(num_channels, kernel_size=1)
        self.avg_pool = AveragePooling2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)


def _db1():
    net = _b1()

    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):

        net.add(DenseBlock(num_convs, growth_rate))
        num_channels += num_convs * growth_rate

        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))

    return net


def build_densenet():

    return Sequential(
        [*_db1().layers, BatchNormalization(), ReLU(), GlobalAvgPool2D(), Flatten(), Dense(10)]
    )


"""SENET"""


class SqueezeExcite(tf.keras.layers.Layer):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    "translated from pytorch to tensorflow"

    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = GlobalAvgPool2D()
        self.excitation = Sequential(
            [
                Dense(c // r, use_bias=False),
                ReLU(),
                Dense(c, use_bias=False),
                Activation("sigmoid"),
            ]
        )

    def call(self, x):
        *_, c = x.shape

        y = self.squeeze(x)
        y = self.excitation(y)

        # print(y.shape)
        # quit()

        y = tf.reshape(y, [-1, 1, 1, c])
        return x * y


class SEResidual(tf.keras.Model):
    """implements SE architecture on ResidualBlock"""

    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()

        self.conv1 = Conv2D(
            num_channels, padding="same", kernel_size=3, strides=strides
        )
        self.conv2 = Conv2D(num_channels, kernel_size=3, padding="same")
        self.conv3 = None

        if use_1x1conv:
            self.conv3 = Conv2D(num_channels, kernel_size=1, strides=strides)

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

        self.se = SqueezeExcite(num_channels)

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        Y = self.se(Y)

        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    SEResidual(num_channels, use_1x1conv=True, strides=2)
                )
            else:
                self.residual_layers.append(SEResidual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X


def build_senet():
    return build_resnet(use_se=True)


"""OTHER"""


def get_shapes(net):
    "prints output shapes to command line"

    X = tf.random.uniform((1, 28, 28, 1))
    for blk in net.layers:
        X = blk(X)
        print(blk.__class__.__name__, "output shape:\t", X.shape)


def test_run(net):
    X = tf.random.uniform((1, 28, 28, 1))

    X = net(X)
    print(X)
    print(f'output shape: {X.shape}')
    print(f'argmax X: {np.argmax(X)}')


def main():
    """vgg-11"""

    model = build_vgg()

    # seres = build_resnet(use_se=True)

    test_run(model)
    get_shapes(model)

    # seres.summary()


if __name__ == "__main__":
    main()
