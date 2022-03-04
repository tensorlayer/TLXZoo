import tensorlayerx as tlx
from ..config import BaseDataConfig
from .task_schema import image_classification_task_data_set_schema
from ..task import BaseForImageClassification
from ..utils.registry import Registers
from .data_random import DataRandom


@Registers.data_configs.register
class ImageClassificationDataConfig(BaseDataConfig):
    task = BaseForImageClassification
    schema = image_classification_task_data_set_schema

    def __init__(self,
                 per_device_train_batch_size=2,
                 per_device_eval_batch_size=2,
                 data_name="Mnist",
                 random_rotation_degrees=15,
                 random_shift=(0.1, 0.1),
                 random_flip_horizontal_prop=0.5,
                 random_crop_size=(32, 4),
                 **kwargs):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.data_name = data_name
        self.task_type = self.task.task_type
        self.random_rotation_degrees = random_rotation_degrees
        self.random_shift = random_shift
        self.random_flip_horizontal_prop = random_flip_horizontal_prop
        self.random_crop_size = random_crop_size
        super(ImageClassificationDataConfig, self).__init__(**kwargs)


# _configs = {BaseForImageClassification.task_type: ImageClassificationDataConfig}


class DataLoaders(object):
    def __init__(self, config, train_limit=None):
        self.config = config
        self.dataset_dict = Registers.datasets[self.config.data_name].load(train_limit)

        get_schema_dataset_func = getattr(self.dataset_dict, f"get_{self.config.task_type}_schema_dataset")

        data_random_hook = DataRandom(random_rotation_degrees=self.config.random_rotation_degrees,
                                      random_shift=self.config.random_shift,
                                      random_flip_horizontal_prop=self.config.random_flip_horizontal_prop,
                                      random_crop_size=self.config.random_crop_size)

        if "train" in self.dataset_dict:
            train_data = get_schema_dataset_func("train")
            train_data.register_random_transform_hook(data_random_hook)
            self.train = self.dataset_dataloader(train_data, dataset_type="train")
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

    def register_feature_transform_hook(self, feature_transform_hook):
        get_schema_dataset_func = getattr(self.dataset_dict, f"get_{self.config.task_type}_schema_dataset")

        if "train" in self.dataset_dict:
            get_schema_dataset_func("train").register_feature_transform_hook(feature_transform_hook)

        if "eval" in self.dataset_dict:
            get_schema_dataset_func("eval").register_feature_transform_hook(feature_transform_hook)

        if "test" in self.dataset_dict:
            get_schema_dataset_func("test").register_feature_transform_hook(feature_transform_hook)

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            if pretrained_path is None:
                raise ValueError("pretrained_path and config are both None.")

            config_dict = BaseDataConfig.get_config_dict_from_path(pretrained_path)
            # config = _configs[config_dict.task_type].from_dict(config_dict)
            config = Registers.data_configs[config_dict.config_class].from_dict(config_dict)

        return cls(config)

    def dataset_dataloader(self, dataset, dataset_type="train", num_workers=8):
        # validate
        dataset.validate(self.config.schema)

        # output_types = self.config.schema.get_dtypes()
        # column_names = self.config.schema.get_names()
        # dataset = tlx.dataflow.FromGenerator(
        #     dataset, output_types=output_types, column_names=column_names
        # )

        if dataset_type == "train":
            train_loader = tlx.dataflow.DataLoader(dataset,
                                                   batch_size=self.config.per_device_train_batch_size,
                                                   prefetch_factor=self.config.per_device_train_batch_size,
                                                   num_workers=num_workers,
                                                   shuffle=True)
            return train_loader
        else:
            return tlx.dataflow.DataLoader(dataset, batch_size=self.config.per_device_eval_batch_size, shuffle=False)
