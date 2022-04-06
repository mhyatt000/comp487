import functools
import os
import time
from argparse import ArgumentParser

import torch
import torchvision
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_args():

    ap = ArgumentParser(description='args for hw4')

    ap.add_argument('-s', '--save', action='store_true')
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('-t', '--train', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('-e', '--epochs', type=int)

    args = ap.parse_args()

    if not args.epochs:
        args.epochs = 5

    return args

class Environment():
    'environment variables to be passed around'

    def __init__(self):
        pass


def get_voc2012():
    d2l.DATA_HUB["voc2012"] = (
        d2l.DATA_URL + "VOCtrainval_11-May-2012.tar",
        "4e443f8a2eca6b1dac8a6c57641b67dd40621a49",
    )


def get_voc_maps():

    VOC_COLORMAP = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
        [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
        [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
    ]
    VOC_CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "potted plant", "sheep", "sofa", "train", "tv/monitor"
    ]
    return VOC_COLORMAP, VOC_CLASSES


def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""

    txt_fname = os.path.join(
        voc_dir, "ImageSets", "Segmentation", "train.txt" if is_train else "val.txt"
    )
    mode = torchvision.io.image.ImageReadMode.RGB

    with open(txt_fname, "r") as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, "JPEGImages", f"{fname}.jpg")
            )
        )
        labels.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, "SegmentationClass", f"{fname}.png"), mode
            )
        )
    return features, labels


def bilinear_kernel(in_channels, out_channels, kernel_size):

    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (
        torch.arange(kernel_size).reshape(-1, 1),
        torch.arange(kernel_size).reshape(1, -1),
    )
    filt = (1 - torch.abs(og[0] - center) / factor) * (
        1 - torch.abs(og[1] - center) / factor
    )
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


def build_model(num_classes=21):

    pretrained_net = torchvision.models.resnet18(pretrained=True)

    net = nn.Sequential(*list(pretrained_net.children())[:-2])

    net.add_module("final_conv", nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module("transpose_conv", nn.ConvTranspose2d(
        num_classes, num_classes, kernel_size=64, padding=16, stride=32))

    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)

    return net


def calc_loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction="none").mean(1).mean(1)


def predict(net, test_iter, img, *, env):

    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(env.device)).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred, *, env):

    colormap = torch.tensor(d2l.VOC_COLORMAP, device=env.device)
    X = pred.long()
    return colormap[X, :]

def train(net, train_iter, optimizer, *, env):

    timer = time.perf_counter()

    # animator = d2l.Animator(xlabel='epoch', xlim=[1, epochs], legend=['pixelwise loss'])

    epochs = env.args.epochs

    device = env.device
    net.to(device)

    losses = []
    for epoch in range(epochs):
        print(f'\nepoch: {epoch+1} of {epochs}')

        net.train()
        for (features, target) in tqdm(train_iter):

            optimizer.zero_grad()
            X, Y = features.to(device), target.to(device)

            fx = net(X)
            loss = calc_loss(fx, Y)
            loss.mean().backward()
            optimizer.step()

            losses.append(loss.mean())

            # animator.add(epoch + 1, (loss.mean()))

            # save it ... after batch cuz it takes a while
            if env.args.save:
                torch.save(net.state_dict(), env.file)

        print(f'  pixelwise loss: {loss.mean()}')

    timer = int(time.perf_counter() - timer)
    print(f'Finished in {timer} seconds')
    
    print(f'{len(train_iter.dataset) / timer:.1f} examples/sec on {str(device)}')


def main():

    args = get_args()

    env = Environment()
    env.file = __file__.split('/')[-1].split('.')[0] + '.pt'
    env.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env.args = args

    # define maps
    VOC_COLORMAP, VOC_CLASSES = get_voc_maps()

    # batch data for training
    print('getting training data...')
    batch_size, crop_size = 32, (320, 480)
    train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

    # create a model
    net = build_model()
    if args.load:
        net.load_state_dict(torch.load(env.file))
    if args.verbose:
        summary(net, (3,*crop_size))

    # train
    lr, wd = 0.001, 1e-3
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

    if args.train:
        train(net, train_iter, optimizer, env=env)

    # do some predictions
    voc_dir = d2l.download_extract("voc2012", "VOCdevkit/VOC2012")
    test_images, test_labels = d2l.read_voc_images(voc_dir, False)

    print('evaluation')
    n, imgs = 4, []
    for i in tqdm(range(n)):
        crop_rect = (0, 0, 320, 480)
        crop = lambda imgs: torchvision.transforms.functional.crop(imgs, *crop_rect)

        X = crop(test_images[i])
        pred = label2image(predict(net, test_iter, X, env=env), env=env)
        imgs += [X.permute(1, 2, 0), pred.cpu(), crop(test_labels[i]).permute(1, 2, 0)]

    d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
    # plt.pause(2)
    plt.show()

if __name__ == '__main__':
    main()
