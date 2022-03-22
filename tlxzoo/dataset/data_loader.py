import tensorlayerx as tlx
from ..config import BaseDataConfig
from .task_schema import image_classification_task_data_set_schema, object_detection_task_data_set_schema
from ..task import *
from ..utils.registry import Registers
import numpy as np
from .dataset import IterableDataset


@Registers.data_configs.register
class ImageClassificationDataConfig(BaseDataConfig):
    task = BaseForImageClassification
    schema = image_classification_task_data_set_schema

    def __init__(self,
                 per_device_train_batch_size=2,
                 per_device_eval_batch_size=2,
                 data_name="Cifar10",
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


@Registers.data_configs.register
class ImageDetectionDataConfig(BaseDataConfig):
    task = BaseForObjectDetection
    schema = object_detection_task_data_set_schema

    def __init__(self,
                 per_device_train_batch_size=2,
                 per_device_eval_batch_size=2,
                 data_name="Coco",
                 train_ann_path="",
                 val_ann_path="",
                 strides=(8, 16, 32),
                 anchors=(12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401),
                 num_classes=80,
                 anchor_per_scale=3,
                 max_bbox_per_scale=150,
                 train_output_sizes=(52, 26, 13),
                 train_input_size=416,
                 **kwargs):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.data_name = data_name
        self.task_type = self.task.task_type
        self.train_ann_path = train_ann_path
        self.val_ann_path = val_ann_path
        self.strides = np.array(strides)
        self.anchors = np.array(anchors).reshape(3, 3, 2)
        self.num_classes = num_classes
        self.anchor_per_scale = anchor_per_scale
        self.max_bbox_per_scale = max_bbox_per_scale
        self.train_output_sizes = np.array(train_output_sizes)
        self.train_input_size = train_input_size
        super(ImageDetectionDataConfig, self).__init__(**kwargs)


@Registers.data_configs.register
class ConditionalGeneration(BaseDataConfig):
    task = BaseForConditionalGeneration
    schema = None

    def __init__(self,
                 per_device_train_batch_size=2,
                 per_device_eval_batch_size=2,
                 data_name="WmtEnfr",
                 **kwargs):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.data_name = data_name
        self.task_type = self.task.task_type
        super(ConditionalGeneration, self).__init__(**kwargs)


@Registers.data_configs.register
class TextClassificationDataConfig(BaseDataConfig):
    task = BaseForTextClassification
    schema = None

    def __init__(self,
                 per_device_train_batch_size=2,
                 per_device_eval_batch_size=2,
                 data_name="SST-2",
                 **kwargs):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.data_name = data_name
        self.task_type = self.task.task_type
        super(TextClassificationDataConfig, self).__init__(**kwargs)


@Registers.data_configs.register
class PairTextClassificationDataConfig(BaseDataConfig):
    task = BaseForPairTextClassification
    schema = None

    def __init__(self,
                 per_device_train_batch_size=2,
                 per_device_eval_batch_size=2,
                 data_name="QQP",
                 **kwargs):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.data_name = data_name
        self.task_type = self.task.task_type
        super(PairTextClassificationDataConfig, self).__init__(**kwargs)

# _configs = {BaseForImageClassification.task_type: ImageClassificationDataConfig}


class DataLoaders(object):
    def __init__(self, config, train_limit=None, collate_fn=None):
        self.config = config
        self.dataset_dict = Registers.datasets[self.config.data_name].load(train_limit, config=self.config)

        get_schema_dataset_func = getattr(self.dataset_dict, f"get_{self.config.task_type}_schema_dataset")

        if "train" in self.dataset_dict:
            train_data = get_schema_dataset_func("train", config)
            self.train = self.dataset_dataloader(train_data, dataset_type="train", collate_fn=collate_fn,
                                                 num_workers=config.num_workers)
        else:
            self.train = None

        if "eval" in self.dataset_dict:
            self.eval = self.dataset_dataloader(get_schema_dataset_func("eval", config), collate_fn=collate_fn,
                                                dataset_type="eval")
        else:
            self.eval = None

        if "test" in self.dataset_dict:
            self.test = self.dataset_dataloader(get_schema_dataset_func("test", config), collate_fn=collate_fn,
                                                dataset_type="test")
        else:
            self.test = None

    def register_transform_hook(self, transform_hook, index=None):
        get_schema_dataset_func = getattr(self.dataset_dict, f"get_{self.config.task_type}_schema_dataset")

        if "train" in self.dataset_dict:
            train_data = get_schema_dataset_func("train", self.config)
            train_data.register_transform_hook(transform_hook, index=index)

        if "eval" in self.dataset_dict:
            get_schema_dataset_func("eval").register_transform_hook(transform_hook, index=index)

        if "test" in self.dataset_dict:
            get_schema_dataset_func("test").register_transform_hook(transform_hook, index=index)

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            if pretrained_path is None:
                raise ValueError("pretrained_path and config are both None.")

            config_dict = BaseDataConfig.get_config_dict_from_path(pretrained_path)
            config = Registers.data_configs[config_dict.config_class].from_dict(config_dict)

        return cls(config)

    def dataset_dataloader(self, dataset, dataset_type="train", num_workers=8, collate_fn=None):
        # validate
        if self.config.schema is not None:
            dataset.validate(self.config.schema)

        # output_types = self.config.schema.get_dtypes()
        # column_names = self.config.schema.get_names()
        # dataset = tlx.dataflow.FromGenerator(
        #     dataset, output_types=output_types, column_names=column_names
        # )

        if dataset_type == "train":
            if isinstance(dataset, IterableDataset):
                shuffle = False
            else:
                shuffle = True
            if num_workers == 0:
                train_loader = tlx.dataflow.DataLoader(dataset,
                                                       batch_size=self.config.per_device_train_batch_size,
                                                       collate_fn=collate_fn,
                                                       shuffle=shuffle)
            else:
                train_loader = tlx.dataflow.DataLoader(dataset,
                                                       batch_size=self.config.per_device_train_batch_size,
                                                       prefetch_factor=self.config.per_device_train_batch_size,
                                                       num_workers=num_workers,
                                                       collate_fn=collate_fn,
                                                       shuffle=shuffle)
            return train_loader
        else:
            return tlx.dataflow.DataLoader(dataset, collate_fn=collate_fn,
                                           batch_size=self.config.per_device_eval_batch_size, shuffle=False)
