from tensorlayerx.files.utils import maybe_download_and_extract
from ...utils.registry import Registers
from ..dataset import BaseDataSetDict, logging, BaseDataSet

import zipfile
import os


@Registers.datasets.register("SST-2")
class SST2DataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, path, train_limit=None):

        maybe_download_and_extract("SST-2.zip", path, "https://dl.fbaipublicfiles.com/glue/data/")

        def unzip_file(gz_path, new_path):
            """Unzips from gz_path into new_path."""
            logging.info("Unpacking %s to %s" % (gz_path, new_path))
            zFile = zipfile.ZipFile(gz_path, "r")
            for fileM in zFile.namelist():
                zFile.extract(fileM, new_path)
            zFile.close()

        unzip_file(os.path.join(path, "SST-2.zip"), path)

        def transformer(path):
            texts = []
            labels = []
            for index, i in enumerate(open(path)):
                if index == 0:
                    continue
                i = i.strip().rsplit(" ", 1)
                texts.append(i[0])
                labels.append(int(i[1]))
            return texts, labels

        train_texts, train_labels = transformer(os.path.join(path, "./SST-2/train.tsv"))
        test_texts, test_labels = transformer(os.path.join(path, "./SST-2/dev.tsv"))

        return cls({"train": BaseDataSet(train_texts, train_labels),
                    "test": BaseDataSet(test_texts, test_labels)})

