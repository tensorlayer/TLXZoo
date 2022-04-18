from ...utils.registry import Registers
from ..dataset import BaseDataSetDict, Dataset, BaseDataSetMixin
import os


class Synth90k(Dataset, BaseDataSetMixin):
    def __init__(
            self, archive_path, label_path, transforms=None, limit=None,
    ):
        self.archive_path = archive_path
        self.label_path = label_path
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []

        super(Synth90k, self).__init__()

        transcripts_file = os.path.join(archive_path, label_path)

        files = []
        for i in open(transcripts_file):
            i = i.strip().split(" ")
            text = i[0].split("_")[1]
            files.append((i[0], text))

        if limit:
            files = files[:limit]

        self.files = files

    def __getitem__(self, index: int):
        jpg_index, text = self.files[index]
        jpg_path = os.path.join(self.archive_path, jpg_index)

        image, target = self.transform(jpg_path, text)

        return image, (target, text)

    def __len__(self) -> int:
        return len(self.files)


@Registers.datasets.register("synth90k")
class IAMDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, config=None):

        return cls({"train": Synth90k(config.root_path, config.train_ann_path, limit=train_limit),
                    "test": Synth90k(config.root_path, config.val_ann_path)})

    def get_automatic_speech_recognition_schema_dataset(self, dataset_type, config=None):

        if dataset_type == "train":
            dataset = self["train"]
        else:
            dataset = self["test"]

        return dataset
