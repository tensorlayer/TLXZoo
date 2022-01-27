import tensorlayerx as tlx
from ..config import BaseDataConfig
from .task_schema import image_classification_task_data_set_schema
from ..task import BaseForImageClassification
from ..utils.registry import Registers


class ImageClassificationDataConfig(BaseDataConfig):
    task = BaseForImageClassification

    def __init__(self,
                 per_device_train_batch_size=2,
                 per_device_eval_batch_size=2,
                 data_name="Mnist",
                 **kwargs):
        self.schema = image_classification_task_data_set_schema
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.data_name = data_name
        self.task_type = self.task.task_type
        super(ImageClassificationDataConfig, self).__init__(**kwargs)


_configs = {BaseForImageClassification.task_type: ImageClassificationDataConfig}


class DataLoaders(object):
    def __init__(self, config):
        self.config = config
        self.dataset_dict = Registers.datasets[self.config.data_name].load()

        get_schema_dataset_func = getattr(self.dataset_dict, f"get_{self.config.task_type}_schema_dataset")

        if "train" in self.dataset_dict:
            self.train = self.dataset_dataloader(get_schema_dataset_func("train"), dataset_type="train")
        else:
            self.train = None

        if "eval" in self.dataset_dict:
            self.eval = self.dataset_dataloader(get_schema_dataset_func("eval"), dataset_type="eval")
        else:
            self.eval = None

        if "test" in self.dataset_dict:
            self.test = self.dataset_dataloader(get_schema_dataset_func("test"), dataset_type="test")
        else:
            self.test = None

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            if pretrained_path is None:
                raise ValueError("pretrained_path and config are both None.")

            config_dict = BaseDataConfig.get_config_dict_from_path(pretrained_path)
            config = _configs[config_dict.task_type].from_dict(config_dict)

        return cls(config)

    def dataset_dataloader(self, dataset, dataset_type="train"):
        # validate
        dataset.validate(self.config.schema)

        output_types = self.config.schema.get_dtypes()
        column_names = self.config.schema.get_names()
        dataset = tlx.dataflow.FromGenerator(
            dataset, output_types=output_types, column_names=column_names
        )

        if dataset_type == "train":
            train_loader = tlx.dataflow.Dataloader(dataset,
                                                   batch_size=self.config.per_device_train_batch_size,
                                                   shuffle=True)
            return train_loader
        else:
            return tlx.dataflow.Dataloader(dataset, batch_size=self.config.per_device_eval_batch_size, shuffle=False)
