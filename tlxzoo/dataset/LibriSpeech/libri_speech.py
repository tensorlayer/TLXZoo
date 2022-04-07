from ...utils.registry import Registers
from ..dataset import BaseDataSetDict, Dataset, BaseDataSetMixin
import glob
import os


class LibriSpeech(Dataset, BaseDataSetMixin):
    def __init__(
            self, archive_path, transforms=None, limit=None,
    ):
        self.archive_path = archive_path
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []

        super(LibriSpeech, self).__init__()

        transcripts_glob = os.path.join(archive_path, "*/*/*.txt")
        files = []
        texts = []

        for transcript_file in sorted(glob.glob(transcripts_glob)):
            path = os.path.dirname(transcript_file)
            with open(transcript_file) as f:
                for line in f:
                    line = line.strip()
                    key, transcript = line.split(" ", 1)

                    audio_file = f"{key}.flac"
                    file = os.path.join(path, audio_file)

                    files.append(file)
                    texts.append(transcript)
        if limit:
            files = files[:limit]
            texts = texts[:limit]

        self.files = files
        self.texts = texts

    def __getitem__(self, index: int):
        import soundfile as sf
        file = self.files[index]
        text = self.texts[index]

        speech, _ = sf.read(file)

        image, target = self.transform(speech, text)

        return image, (target, text)

    def __len__(self) -> int:
        return len(self.files)


@Registers.datasets.register("LibriSpeech")
class LibriSpeechDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, config=None):

        return cls({"train": LibriSpeech(config.train_path, limit=train_limit),
                    "test": LibriSpeech(config.test_path)})

    def get_automatic_speech_recognition_schema_dataset(self, dataset_type, config=None):

        if dataset_type == "train":
            dataset = self["train"]
        else:
            dataset = self["test"]

        return dataset



