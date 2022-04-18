from ...task.task import BaseForHumanPoseEstimation
from ...utils.output import BaseTaskOutput
from .h import *
import numpy as np
import tensorlayerx as tlx


class HRNetForHumanPoseEstimation(BaseForHumanPoseEstimation):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.model = PoseHighResolutionNet(config.model_config)

    def forward(self, inputs):
        logits = self.model(inputs)
        return BaseTaskOutput(logits=logits)

    def loss_fn(self, y_pred, target, target_weight):
        mse = tlx.losses.mean_squared_error

        batch_size = y_pred.shape[0]
        num_of_joints = y_pred.shape[-1]

        pred = tlx.reshape(tensor=y_pred, shape=(batch_size, -1, num_of_joints))
        gt = tlx.reshape(tensor=target, shape=(batch_size, -1, num_of_joints))

        loss = 0
        for i in range(num_of_joints):
            heatmap_pred = pred[:, :, i]
            heatmap_gt = gt[:, :, i]
            loss += 0.5 * mse(target=heatmap_pred * target_weight[:, i],
                              output=heatmap_gt * target_weight[:, i])
        bloss = loss / num_of_joints
        return bloss
        # import tensorflow as tf
        # mse = tf.losses.MeanSquaredError()
        # batch_size = y_pred.shape[0]
        # num_of_joints = y_pred.shape[-1]
        # pred = tf.reshape(tensor=y_pred, shape=(batch_size, -1, num_of_joints))
        # heatmap_pred_list = tf.split(value=pred, num_or_size_splits=num_of_joints, axis=-1)
        # gt = tf.reshape(tensor=target, shape=(batch_size, -1, num_of_joints))
        # heatmap_gt_list = tf.split(value=gt, num_or_size_splits=num_of_joints, axis=-1)
        # loss = 0.0
        # for i in range(num_of_joints):
        #     heatmap_pred = tf.squeeze(heatmap_pred_list[i])
        #     heatmap_gt = tf.squeeze(heatmap_gt_list[i])
        #     loss += 0.5 * mse(y_true=heatmap_pred * target_weight[:, i],
        #                            y_pred=heatmap_gt * target_weight[:, i])
        # return loss / num_of_joints


def get_max_preds(heatmap_tensor):
    heatmap = tlx.convert_to_numpy(heatmap_tensor)
    batch_size, _, width, num_of_joints = heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[-1]
    heatmap = heatmap.reshape((batch_size, -1, num_of_joints))
    index = np.argmax(heatmap, axis=1)
    maxval = np.amax(heatmap, axis=1)
    index = index.reshape((batch_size, 1, num_of_joints))
    maxval = maxval.reshape((batch_size, 1, num_of_joints))
    preds = np.tile(index, (1, 2, 1)).astype(np.float32)

    preds[:, 0, :] = preds[:, 0, :] % width
    preds[:, 1, :] = np.floor(preds[:, 1, :] / width)

    pred_mask = np.tile(np.greater(maxval, 0.0), (1, 2, 1))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask

    return preds, maxval


class PCK(object):
    def __init__(self):
        self.threshold = 0.5

    def __call__(self, network_output, target):
        _, h, w, c = network_output.shape
        index = list(range(c))
        pred, _ = get_max_preds(heatmap_tensor=network_output)
        target, _ = get_max_preds(heatmap_tensor=target)
        normalize = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        distance = self.__calculate_distance(pred, target, normalize)

        accuracy = np.zeros((len(index) + 1))
        average_accuracy = 0
        count = 0

        for i in range(c):
            accuracy[i + 1] = self.__distance_accuracy(distance[index[i]])
            if accuracy[i + 1] > 0:
                average_accuracy += accuracy[i + 1]
                count += 1
        average_accuracy = average_accuracy / count if count != 0 else 0
        if count != 0:
            accuracy[0] = average_accuracy
        return accuracy, average_accuracy, count, pred

    @staticmethod
    def __calculate_distance(pred, target, normalize):
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
        distance = np.zeros((pred.shape[-1], pred.shape[0]))
        for n in range(pred.shape[0]):
            for c in range(pred.shape[-1]):
                if target[n, 0, c] > 1 and target[n, 1, c] > 1:
                    normed_preds = pred[n, :, c] / normalize[n]
                    normed_targets = target[n, :, c] / normalize[n]
                    distance[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    distance[c, n] = -1
        return distance

    def __distance_accuracy(self, distance):
        distance_calculated = np.not_equal(distance, -1)
        num_dist_cal = distance_calculated.sum()
        if num_dist_cal > 0:
            return np.less(distance[distance_calculated], self.threshold).sum() * 1.0 / num_dist_cal
        else:
            return -1