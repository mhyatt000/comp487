import os
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def get_voc2012():
    d2l.DATA_HUB["voc2012"] = (
        d2l.DATA_URL + "VOCtrainval_11-May-2012.tar",
        "4e443f8a2eca6b1dac8a6c57641b67dd40621a49",
    )


voc_dir = d2l.download_extract("voc2012", "VOCdevkit/VOC2012")


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


train_features, train_labels = read_voc_images(voc_dir, True)

n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs, 2, n)

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


pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]

net = nn.Sequential(*list(pretrained_net.children())[:-2])

num_classes = 21
net.add_module("final_conv", nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module(
    "transpose_conv",
    nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32),
)


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


W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)


def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction="none").mean(1).mean(1)


num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]


voc_dir = d2l.download_extract("voc2012", "VOCdevkit/VOC2012")
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [
        X.permute(1, 2, 0),
        pred.cpu(),
        torchvision.transforms.functional.crop(test_labels[i], *crop_rect).permute(1, 2, 0),
    ]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
