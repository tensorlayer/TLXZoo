from tensorlayerx import logging
import abc
import json
import os
import copy
from ..utils import MODEL_CONFIG, TASK_CONFIG, IMAGE_FEATURE_CONFIG

_config_type_name = {"": "",
                     "model": MODEL_CONFIG,
                     "task": TASK_CONFIG,
                     "image_feature": IMAGE_FEATURE_CONFIG
                     }


class PreTrainedMixin:
    """
    A few utilities for pretrain, to be used as a mixin.
    """


class BaseConfig(object):
    config_type = "base"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logging.error(f"Can't set {key} with value {value} for {self}")
                raise err

    @classmethod
    def _default_cls(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    def _from_dict(cls, config_dict):
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict):
        return cls._from_dict(config_dict)

    @classmethod
    def get_config_dict_from_path(cls, path):
        return json.load(open(path))

    @classmethod
    def from_pretrained(cls, config_path, **kwargs):
        if config_path is None:
            return cls._default_cls(**kwargs)

        if os.path.isdir(config_path):
            config_path = os.path.join(config_path, _config_type_name[cls.config_type])

        config_dict = cls.get_config_dict_from_path(config_path)
        if config_dict["config_type"] != cls.config_type:
            raise ValueError(f'{config_dict["config_type"]} is not same as {cls.config_type}')

        config_dict["base_path"] = os.path.dirname(config_path)

        return cls._from_dict(config_dict)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self._save_sub_pretrained(save_directory)

        _dict = self.to_dict()
        config_file_path = os.path.join(save_directory, _config_type_name[self.config_type])
        json.dump(_dict, open(config_file_path, "w"))
        return config_file_path

    def _save_sub_pretrained(self, save_directory):
        ...

    def _post_dict(self, _dict):
        return _dict

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["config_type"] = self.config_type
        return self._post_dict(output)

    def __eq__(self, other):
        self_dict = self.to_dict()
        if "base_path" in self_dict:
            del self_dict["base_path"]
        other_dict = other.to_dict()
        if "base_path" in other_dict:
            del other_dict["base_path"]
        return self_dict == other_dict


class BaseModelConfig(BaseConfig):
    config_type = "model"

    def get_last_output_size(self):
        size = self._get_last_output_size()
        if not isinstance(size, tuple):
            raise TypeError(f"Type of size is tuple, get {type(size)}")
        return size

    def _get_last_output_size(self):
        raise NotImplementedError

    @classmethod
    def _default_cls(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    def _from_dict(cls, config_dict):
        return cls(**config_dict)


class BaseTaskConfig(BaseConfig):
    config_type = "task"
    model_config_type = BaseModelConfig

    def __init__(self, model_config: BaseModelConfig, **kwargs):
        self.model_config = model_config
        self.model_config_path = kwargs.pop("model_config_path", MODEL_CONFIG)
        super(BaseTaskConfig, self).__init__(**kwargs)

    @classmethod
    def _default_cls(cls, **kwargs):
        model_config = cls.model_config_type(**kwargs)
        return cls(model_config, **kwargs)

    @classmethod
    def _from_dict(cls, config_dict):
        model_config = cls.model_config_type.from_pretrained(
            os.path.join(config_dict["base_path"], config_dict["model_config_path"]))
        return cls(model_config, **config_dict)

    def _post_dict(self, _dict):
        del _dict["model_config"]
        _dict["model_config_path"] = MODEL_CONFIG
        return _dict

    def _save_sub_pretrained(self, save_directory):
        self.model_config.save_pretrained(save_directory)


class BaseFeatureConfig(BaseConfig):
    config_type = "feature"

    @classmethod
    def _default_cls(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    def _from_dict(cls, config_dict):
        return cls(**config_dict)


class BaseImageFeatureConfig(BaseFeatureConfig):
    config_type = "image_feature"

    def __init__(self, do_resize=True,
                 do_normalize=True,
                 resize_size=(224, 224),
                 mean=(123.68, 116.779, 103.939),
                 std=None,
                 **kwargs):
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.resize_size = tuple(resize_size)
        self.mean = list(mean)
        self.std = std
        super(BaseImageFeatureConfig, self).__init__(**kwargs)


class BaseTextFeatureConfig(BaseFeatureConfig):
    config_type = "text_feature"


class BaseInferConfig(BaseConfig):
    config_type = "infer"


class BaseDataConfig(BaseConfig):
    config_type = "data"


class BaseRunnerConfig(BaseConfig):
    config_type = "runner"


class BaseAppConfig(BaseConfig):
    config_type = "app"
