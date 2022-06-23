import numpy as np
import tensorlayerx as tlx


class NME(object):
    def __init__(self, dist='ION', npoints=68):
        assert dist in ['ION', 'IPN'], 'Invalid dist'
        assert npoints in [68], 'Invalid point num'

        self.dist = dist
        self.npoints = npoints
        self.sum = 0.0
        self.num = 0

    def update(self, pred, target):
        if type(pred) == tuple:
            pred = pred[0]
            target = target[0]

        batch_size = len(pred)
        pred = np.array(pred).reshape((batch_size, -1, 2))
        target = np.array(target).reshape((batch_size, -1, 2))

        if self.dist == 'ION':
            if self.npoints == 68:
                d = np.linalg.norm(target[:, 36, :] - target[:, 45, :], axis=1)
        if self.dist == 'IPN':
            if self.npoints == 68:
                left = tlx.reduce_mean(
                    target[:, [36, 37, 38, 39, 40, 41], :], axis=1)
                right = tlx.reduce_mean(
                    target[:, [42, 43, 44, 45, 46, 47], :], axis=1)
                d = np.linalg.norm(left - right, axis=1)

        self.sum += np.sum(np.sum(np.linalg.norm(pred - target,
                           axis=2), axis=1) / d / self.npoints)
        self.num += batch_size

    def reset(self):
        self.sum = 0.0
        self.num = 0

    def result(self):
        return self.sum / self.num
