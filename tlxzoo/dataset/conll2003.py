from .dataset import *
from tensorlayerx.files.utils import maybe_download_and_extract
import zipfile
import os


@Registers.datasets.register("Conll2003")
class Conll2003DataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, config=None):
        maybe_download_and_extract("conll2003.zip", config.path, "https://data.deepai.org/")

        def unzip_file(gz_path, new_path):
            """Unzips from gz_path into new_path."""
            logging.info("Unpacking %s to %s" % (gz_path, new_path))
            zFile = zipfile.ZipFile(gz_path, "r")
            for fileM in zFile.namelist():
                zFile.extract(fileM, new_path)
            zFile.close()

        unzip_file(os.path.join(config.path, "conll2003.zip"), os.path.join(config.path, "conll2003"))

        tags_set = set()
        with open(os.path.join(config.path, "conll2003/valid.txt"), encoding="utf-8") as f:
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    ...
                else:
                    splits = line.split(" ")
                    if config.task == "pos":
                        tags_set.add(splits[1])
                    elif config.task == "chunk":
                        tags_set.add(splits[2])
                    else:
                        tags_set.add(splits[3].rstrip())

        tags_set = list(tags_set)

        def transformer(path, tags_set):
            texts = []
            labels = []
            tokens = []
            tags = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        if len(tokens) != 0:
                            texts.append(tokens)
                            labels.append(tags)
                        tokens = []
                        tags = []
                    else:
                        splits = line.split(" ")
                        tokens.append(splits[0])
                        if config.task == "pos":
                            tags.append(tags_set.index(splits[1]))
                        elif config.task == "chunk":
                            tags.append(tags_set.index(splits[2]))
                        else:
                            tags.append(tags_set.index(splits[3].rstrip()))

            return texts, labels

        train_texts, train_labels = transformer(os.path.join(config.path, "conll2003/train.txt"), tags_set)
        valid_texts, valid_labels = transformer(os.path.join(config.path, "conll2003/valid.txt"), tags_set)
        test_texts, test_labels = transformer(os.path.join(config.path, "conll2003/test.txt"), tags_set)

        return cls({"train": BaseDataSet(train_texts, train_labels),
                    "val": BaseDataSet(valid_texts, valid_labels),
                    "test": BaseDataSet(test_texts, test_labels)})

    def get_token_classification_schema_dataset(self, dataset_type, config=None):
        if dataset_type == "train":
            dataset = self["train"]
        elif dataset_type == "val":
            dataset = self["val"]
        else:
            dataset = self["test"]

        return dataset