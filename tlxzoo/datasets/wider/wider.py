from ...utils.registry import Registers
from ..dataset import BaseDataSetDict, Dataset, BaseDataSetMixin
import os
import numpy as np


def load_info(txt_path):
    """load info from txt"""
    img_paths = []
    words = []

    f = open(txt_path, 'r')
    lines = f.readlines()
    isFirst = True
    labels = []
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            if isFirst is True:
                isFirst = False
            else:
                labels_copy = labels.copy()
                words.append(labels_copy)
                labels.clear()
            path = line[2:]
            path = txt_path.replace('label.txt', 'images/') + path
            img_paths.append(path)
        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            labels.append(label)

    words.append(labels)
    return img_paths, words


def get_target(labels):
    annotations = np.zeros((0, 15))
    if len(labels) == 0:
        return annotations
    for idx, label in enumerate(labels):
        annotation = np.zeros((1, 15))
        # bbox
        annotation[0, 0] = label[0]  # x1
        annotation[0, 1] = label[1]  # y1
        annotation[0, 2] = label[0] + label[2]  # x2
        annotation[0, 3] = label[1] + label[3]  # y2

        if len(label) > 4:
            # landmarks
            annotation[0, 4] = label[4]  # l0_x
            annotation[0, 5] = label[5]  # l0_y
            annotation[0, 6] = label[7]  # l1_x
            annotation[0, 7] = label[8]  # l1_y
            annotation[0, 8] = label[10]  # l2_x
            annotation[0, 9] = label[11]  # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1  # w/o landmark
            else:
                annotation[0, 14] = 1

        annotations = np.append(annotations, annotation, axis=0)
    target = np.array(annotations)

    return target


class Wider(Dataset, BaseDataSetMixin):
    def __init__(
            self, archive_path, label_path, transforms=None, limit=None,
    ):
        self.archive_path = archive_path
        self.label_path = label_path
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []

        super(Wider, self).__init__()

        transcripts_file = os.path.join(archive_path, label_path)

        img_paths, words = load_info(transcripts_file)

        if limit:
            img_paths = img_paths[:limit]
            words = words[:limit]

        self.img_paths = img_paths
        self.words = words

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]

        word = self.words[index]
        target = get_target(word)

        image, target = self.transform(img_path, target)

        return image, target

    def __len__(self) -> int:
        return len(self.img_paths)


@Registers.datasets.register("wider")
class WiderDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, root_path, train_ann_path, val_ann_path, train_limit=None):

        return cls({"train": Wider(os.path.join(root_path, "train"),
                                   train_ann_path, limit=train_limit),
                    "test": Wider(os.path.join(root_path, "val"), val_ann_path)})
