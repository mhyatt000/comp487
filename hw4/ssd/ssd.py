import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np


def get_args():

    ap = ArgumentParser(description='args for hw4')

    ap.add_argument('-s', '--save', action='store_true')
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('-t', '--train', action='store_true')

    return ap.parse_args()


"""TODO look at 13.4-13.6 for helper methods"""


def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(
        num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1
    )


def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    return block(x)


def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i + 1]))
    return nn.Sequential(*blk)


def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]

        self.sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
        self.ratios = [[1, 2, 0.5]] * 5
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1

        # better with list comprehension instead ... blks = []
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f"blk_{i}", get_blk(i))
            setattr(
                self,
                f"cls_{i}",
                cls_predictor(idx_to_in_channels[i], self.num_anchors, num_classes),
            )
            setattr(
                self, f"bbox_{i}", bbox_predictor(idx_to_in_channels[i], self.num_anchors)
            )

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X,
                getattr(self, f"blk_{i}"),
                self.sizes[i],
                self.ratios[i],
                getattr(self, f"cls_{i}"),
                getattr(self, f"bbox_{i}"),
            )
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    '''calculates 2 part loss ... classification and offset from bbox'''

    cls_loss = nn.CrossEntropyLoss(reduction='none')
    bbox_loss = nn.L1Loss(reduction='none')

    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


def train(device, net, trainer, train_iter):

    args = get_args()

    num_epochs, timer = 20, d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['class error', 'bbox mae'])
    net = net.to(device)
    for epoch in range(num_epochs):
        # Sum of training accuracy, no. of examples in sum of training accuracy,
        # Sum of absolute error, no. of examples in sum of absolute error
        metric = d2l.Accumulator(4)
        net.train()
        for features, target in train_iter:
            timer.start()
            trainer.zero_grad()
            X, Y = features.to(device), target.to(device)
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
            # Calculate the loss function using the predicted and labeled values
            # of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
            l.mean().backward()
            trainer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                       bbox_labels.numel())
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        animator.add(epoch + 1, (cls_err, bbox_mae))

        # save it
    if args.save:
        torch.save(net.state_dict(), './ssd_net.pt')

    print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')


def predict(net, X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X)
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

#
# def box_iou(boxes1, boxes2):
#     """Compute pairwise IoU across two lists of anchor or bounding boxes."""
#     def box_area(boxes):
#         return ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
#     # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
#     # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
#     areas1 = box_area(boxes1)
#     areas2 = box_area(boxes2)
#     # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
#     # boxes1, no. of boxes2, 2)
#     inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
#     inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
#     inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
#     # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
#     inter_areas = inters[:, :, 0] * inters[:, :, 1]
#     union_areas = areas1[:, None] + areas2 - inter_areas
#     return inter_areas / union_areas
#
#
# def offset_inverse(anchors, offset_preds):
#     """Predict bounding boxes based on anchor boxes with predicted offsets."""
#     anc = d2l.box_corner_to_center(anchors)
#     pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
#     pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
#     pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
#     predicted_bbox = d2l.box_center_to_corner(pred_bbox)
#     return predicted_bbox
#
#
# def nms(boxes, scores, iou_threshold):
#     """Sort confidence scores of predicted bounding boxes."""
#     B = torch.argsort(scores, dim=-1, descending=True)
#     keep = []  # Indices of predicted bounding boxes that will be kept
#     while B.numel() > 0:
#         i = B[0]
#         keep.append(i)
#         if B.numel() == 1:
#             break
#         iou = box_iou(boxes[i, :].reshape(-1, 4),
#                       boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
#         inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
#         B = B[inds + 1]
#     return torch.tensor(keep, device=boxes.device)
#
#
# def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
#                        pos_threshold=0.009999999):
#     """Predict bounding boxes using non-maximum suppression."""
#     device, batch_size = cls_probs.device, cls_probs.shape[0]
#     anchors = anchors.squeeze(0)
#     num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[1]
#     out = []
#     for i in range(batch_size):
#         cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
#         conf, class_id = torch.max(cls_prob[1:], 0)
#         predicted_bb = offset_inverse(anchors, offset_pred)
#         keep = nms(predicted_bb, conf, nms_threshold)
#         # Find all non-`keep` indices and set the class to background
#         all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
#         combined = torch.cat((keep, all_idx))
#         uniques, counts = combined.unique(return_counts=True)
#         non_keep = uniques[counts == 1]
#         all_id_sorted = torch.cat((keep, non_keep))
#         class_id[non_keep] = -1
#         class_id = class_id[all_id_sorted]
#         conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
#         # Here `pos_threshold` is a threshold for positive (non-background)
#         # predictions
#         below_min_idx = (conf < pos_threshold)
#         class_id[below_min_idx] = -1
#         conf[below_min_idx] = 1 - conf[below_min_idx]
#         pred_info = torch.cat((class_id.unsqueeze(1),
#                                conf.unsqueeze(1),
#                                predicted_bb), dim=1)
#         out.append(pred_info)
#     return torch.stack(out)


def display(img, output, threshold):

    # output = multibox_detection(output[:, 1], [np.array([[0, 0, 0, 0]]) for i in output],
    #                             output[:, 2:6], nms_threshold=0.5)

    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, f'banana={round(score,2)}', 'r')


def main():

    args = get_args()

    net = TinySSD(num_classes=1)

    if args.load:
        net.load_state_dict(torch.load('./ssd_net.pt'))

    batch_size = 32
    train_iter, _ = d2l.load_data_bananas(batch_size)

    device, net = d2l.try_gpu(), TinySSD(num_classes=1)
    trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    if args.train:
        train(device, net, trainer, train_iter)

    for i in range(100):

        X = torchvision.io.read_image(
            f'../data/banana-detection/bananas_val/images/{i}.png').unsqueeze(0).float()
        img = X.squeeze(0).permute(1, 2, 0).long()

        output = predict(net, X)
        display(img, output.cpu(), threshold=0.8)

        plt.pause(0.25)
        plt.clf()


if __name__ == "__main__":
    main()
