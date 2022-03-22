from .dataset import *
from tensorlayerx.files.utils import maybe_download_and_extract
import zipfile
import os


@Registers.datasets.register("QQP")
class QQPDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, config=None):

        maybe_download_and_extract("QQP-clean.zip", config.path, "https://dl.fbaipublicfiles.com/glue/data/")

        def unzip_file(gz_path, new_path):
            """Unzips from gz_path into new_path."""
            logging.info("Unpacking %s to %s" % (gz_path, new_path))
            zFile = zipfile.ZipFile(gz_path, "r")
            for fileM in zFile.namelist():
                zFile.extract(fileM, new_path)
            zFile.close()

        unzip_file(os.path.join(config.path, "QQP-clean.zip"), config.path)

        def transformer(path):
            texts = []
            labels = []
            for index, i in enumerate(open(path)):
                if index == 0:
                    continue
                i = i.strip().rsplit("	")
                texts.append((i[3], i[4]))
                labels.append(int(i[5]))
            return texts, labels

        train_texts, train_labels = transformer(os.path.join(config.path, "./QQP/train.tsv"))
        test_texts, test_labels = transformer(os.path.join(config.path, "./QQP/dev.tsv"))

        return cls({"train": BaseDataSet(train_texts, train_labels),
                    "test": BaseDataSet(test_texts, test_labels)})

    def get_pair_text_classification_schema_dataset(self, dataset_type, config=None):
        if dataset_type == "train":
            dataset = self["train"]
        else:
            dataset = self["test"]

        return dataset
