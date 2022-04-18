from ...utils.registry import Registers
from ..dataset import BaseDataSetDict, Dataset, BaseDataSetMixin
import glob
import os

# download data from https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database


class IAM(Dataset, BaseDataSetMixin):
    def __init__(
            self, archive_path, label_path, transforms=None, limit=None,
    ):
        self.archive_path = archive_path
        self.label_path = label_path
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []

        super(IAM, self).__init__()

        transcripts_file = os.path.join(archive_path, label_path)

        files = []
        for i in open(transcripts_file):
            if i.startswith("#"):
                continue
            i = i.strip().split(" ")
            text = i[-1]
            text = text.replace("|", " ")
            files.append((i[0], text))

        if limit:
            files = files[:limit]

        self.files = files

    def __getitem__(self, index: int):
        jpg_index, text = self.files[index]
        # synth90k
        if "/" in jpg_index:
            jpg_path = os.path.join(self.archive_path, jpg_index)
        else:
            jpg_index = jpg_index.split("-")
            jpg_path = os.path.join(self.archive_path, f"{jpg_index[0]}", f"{jpg_index[0]}-{jpg_index[1]}",
                                    f"{jpg_index[0]}-{jpg_index[1]}-{jpg_index[2]}.png")

        image, target = self.transform(jpg_path, text)

        return image, (target, text)

    def __len__(self) -> int:
        return len(self.files)


@Registers.datasets.register("IAM")
class IAMDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, config=None):

        return cls({"train": IAM(config.root_path, config.train_ann_path, limit=train_limit),
                    "test": IAM(config.root_path, config.val_ann_path)})

    def get_automatic_speech_recognition_schema_dataset(self, dataset_type, config=None):

        if dataset_type == "train":
            dataset = self["train"]
        else:
            dataset = self["test"]

        return dataset
