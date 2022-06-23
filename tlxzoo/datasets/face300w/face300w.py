import os
import re

import cv2
import numpy as np
from scipy.io import loadmat

from ...utils.registry import Registers
from ..dataset import BaseDataSetDict, BaseDataSetMixin, Dataset


def read_pts_file(path):
    with open(path) as f:
        s = f.read()
    landmarks = re.findall('(\d+.\d+)\s+', s)
    landmarks = [float(v) for v in landmarks]
    landmarks = np.asarray(landmarks).reshape((-1, 2))
    return landmarks


class Face300W(Dataset, BaseDataSetMixin):
    def __init__(self, root_path, image_paths_and_label_files, transforms=None, limit=None):
        self.image_filenames = []
        self.bboxes = []
        self.landmarks = []

        for image_path, label_file in image_paths_and_label_files:
            labels = loadmat(os.path.join(root_path, label_file))[
                'bounding_boxes'][0]
            if 'ibug' in label_file:
                labels = labels[:135]
            for label in labels:
                image_filename = label[0, 0][0][0]
                image_filename = os.path.join(
                    root_path, image_path, image_filename)
                self.image_filenames.append(image_filename)

                bbox = label[0, 0][2][0] - 1
                self.bboxes.append(bbox)

                pts_filename = os.path.splitext(image_filename)[0] + '.pts'
                self.landmarks.append(read_pts_file(pts_filename))

        if limit:
            self.image_filenames = self.image_filenames[:limit]
            self.bboxes = self.bboxes[:limit]
            self.landmarks = self.landmarks[:limit]

        if transforms:
            self.transforms = transforms
        else:
            self.transforms = []

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index])
        bbox = self.bboxes[index]
        landmark = self.landmarks[index]
        return self.transform(image, (bbox, landmark))

    def __len__(self):
        return len(self.image_filenames)


@Registers.datasets.register("Face300W")
class Face300WDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, root_path, train_limit=None):
        return cls({
            "train": Face300W(root_path, [
                ('helen/trainset', 'Bounding Boxes/bounding_boxes_helen_trainset.mat'),
                ('lfpw/trainset', 'Bounding Boxes/bounding_boxes_lfpw_trainset.mat'),
                ('afw', 'Bounding Boxes/bounding_boxes_afw.mat')
            ], limit=train_limit),
            "test": Face300W(root_path, [
                ('helen/testset', 'Bounding Boxes/bounding_boxes_helen_testset.mat'),
                ('lfpw/testset', 'Bounding Boxes/bounding_boxes_lfpw_testset.mat'),
                ('ibug', 'Bounding Boxes/bounding_boxes_ibug.mat')
            ])
        })
