from tlxzoo.module import *
import tensorlayerx as tlx


class ImageSegmentation(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super(ImageSegmentation, self).__init__()
        if backbone == "unet":
            nx = kwargs.pop("nx", 172)
            ny = kwargs.pop("ny", 172)
            channels = kwargs.pop("channels", 1)
            num_classes = kwargs.pop("num_classes", 2)
            layer_depth = kwargs.pop("layer_depth", 3)
            filters_root = kwargs.pop("filters_root", 64)
            self.backbone = Unet(nx=nx, ny=ny, channels=channels, num_classes=num_classes,
                                 layer_depth=layer_depth, filters_root=filters_root)
        else:
            raise ValueError(f"tlxzoo don`t support {backbone}")

    def loss_fn(self, output, target, name="", **kwargs):
        labels_argmax = tlx.argmax(target, -1)
        return tlx.losses.softmax_cross_entropy_with_logits(output, labels_argmax)

    def forward(self, inputs):
        return self.backbone(inputs)

    def predict(self, inputs):
        self.set_eval()
        out = self.backbone(inputs)
        return out


def mean_iou(y_true, y_pred):
    y_true = tlx.cast(y_true, tlx.float64)
    y_pred = tlx.cast(y_pred, tlx.float64)
    I = tlx.reduce_sum(y_pred * y_true, axis=(1, 2))
    U = tlx.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
    return tlx.reduce_mean(I / U)


def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tlx.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tlx.reduce_sum(y_true, axis=[1, 2, 3]) + tlx.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = tlx.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


class Accuracy(tlx.metrics.Accuracy):
    def update(self, y_pred, y_true):
        y_batch_argmax = tlx.argmax(y_true, -1)
        y_batch_argmax = tlx.reshape(y_batch_argmax, [-1])
        _logits = tlx.reshape(y_pred, [-1, tlx.get_tensor_shape(y_pred)[-1]])
        super(Accuracy, self).update(_logits, y_batch_argmax)


def val(model, test_data):
    from tqdm import tqdm
    metrics_acc = tlx.metrics.Accuracy()
    model.set_eval()
    auc_sum = 0
    mean_iou_sum = 0
    dice_coefficient_sum = 0
    num = 0
    for x, labels in tqdm(test_data):
        _logits = model(x)
        _logits = tlx.softmax(_logits)
        mean_iou_sum += mean_iou(labels, _logits)
        dice_coefficient_sum += dice_coefficient(labels, _logits)
        y_batch_argmax = tlx.argmax(labels, -1)
        y_batch_argmax = tlx.reshape(y_batch_argmax, [-1])
        _logits = tlx.reshape(_logits, [-1, tlx.get_tensor_shape(_logits)[-1]])
        metrics_acc.update(_logits, y_batch_argmax)
        auc_sum += metrics_acc.result()
        metrics_acc.reset()
        num += 1

    print(f"val_auc: {auc_sum / num}")
    print(f"val_mean_iou: {mean_iou_sum / num}")
    print(f"val_dice_coefficient: {dice_coefficient_sum / num}")