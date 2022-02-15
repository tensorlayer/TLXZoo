from tensorlayerx import logging
import json
import os
import copy
from ..utils.registry import Registers
from ..utils import MODEL_CONFIG, TASK_CONFIG, FEATURE_CONFIG, TRAINER_CONFIG, DATA_CONFIG, \
    INFER_CONFIG, RUNNER_CONFIG

_config_type_name = {"": "",
                     "model": MODEL_CONFIG,
                     "task": TASK_CONFIG,
                     "feature": FEATURE_CONFIG,
                     "trainer": TRAINER_CONFIG,
                     "data": DATA_CONFIG,
                     "infer": INFER_CONFIG,
                     "runner": RUNNER_CONFIG
                     }

_config_type_register = {
    "model": Registers.model_configs,
    "task": Registers.task_configs,
    "feature": Registers.feature_configs,
    "trainer": Registers.trainer_configs,
    "data": Registers.data_configs,
    "infer": Registers.infer_configs,
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

        if config_dict["config_type"] == "runner":
            return cls._from_dict(config_dict)

        if "config_class" in config_dict:
            return _config_type_register[cls.config_type][config_dict["config_class"]]._from_dict(config_dict)

        return cls._from_dict(config_dict)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self._save_sub_pretrained(save_directory)

        _dict = self.to_dict()
        config_file_path = os.path.join(save_directory, _config_type_name[self.config_type])
        json.dump(_dict, open(config_file_path, "w"), indent=4)
        return config_file_path

    def _save_sub_pretrained(self, save_directory):
        ...

    def _post_dict(self, _dict):
        return _dict

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["config_type"] = self.config_type
        output["config_class"] = self.__class__.__name__
        return self._post_dict(output)

    def __eq__(self, other):
        self_dict = self.to_dict()
        if "base_path" in self_dict:
            del self_dict["base_path"]
        other_dict = other.to_dict()
        if "base_path" in other_dict:
            del other_dict["base_path"]
        return self_dict == other_dict


@Registers.model_configs.register
class BaseModelConfig(BaseConfig):
    config_type = "model"

    def __init__(self, weights_path=None, **kwargs):
        self.weights_path = weights_path
        super(BaseModelConfig, self).__init__(**kwargs)

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


@Registers.task_configs.register
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
        model_config = BaseModelConfig.from_pretrained(
            os.path.join(config_dict["base_path"], config_dict["model_config_path"]))
        return cls(model_config, **config_dict)

    def _post_dict(self, _dict):
        del _dict["model_config"]
        _dict["model_config_path"] = MODEL_CONFIG
        return _dict

    def _save_sub_pretrained(self, save_directory):
        self.model_config.save_pretrained(save_directory)


@Registers.feature_configs.register
class BaseFeatureConfig(BaseConfig):
    config_type = "feature"

    @classmethod
    def _default_cls(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    def _from_dict(cls, config_dict):
        return cls(**config_dict)


@Registers.feature_configs.register
class BaseImageFeatureConfig(BaseFeatureConfig):
    config_type = "feature"

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


@Registers.feature_configs.register
class BaseTextFeatureConfig(BaseFeatureConfig):
    config_type = "feature"


@Registers.infer_configs.register
class BaseInferConfig(BaseConfig):
    config_type = "infer"


@Registers.data_configs.register
class BaseDataConfig(BaseConfig):
    config_type = "data"


@Registers.trainer_configs.register
class BaseTrainerConfig(BaseConfig):
    config_type = "trainer"

    def __init__(self,
                 loss="softmax_cross_entropy_with_logits",
                 optimizers="Momentum",
                 lr=(0.05, 0.9),
                 metric="Accuracy",
                 seed=42,
                 epochs=5,
                 **kwargs):
        self.loss = loss
        self.optimizers = optimizers
        self.lr = lr
        self.metric = metric
        self.seed = seed
        self.epochs = epochs
        super(BaseTrainerConfig, self).__init__(**kwargs)


class BaseRunnerConfig(BaseConfig):
    config_type = "runner"

    def __init__(self,
                 data_config: BaseDataConfig,
                 feature_config: BaseFeatureConfig,
                 task_config: BaseTaskConfig,
                 trainer_config: BaseTrainerConfig,
                 infer_config: BaseInferConfig,
                 **kwargs
                 ):
        self.data_config = data_config
        self.feature_config = feature_config
        self.task_config = task_config
        self.trainer_config = trainer_config
        self.infer_config = infer_config
        super(BaseRunnerConfig, self).__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, config_path, **kwargs):
        if config_path is None:
            raise ValueError("config path is None")
        return super(BaseRunnerConfig, cls).from_pretrained(config_path, **kwargs)

    def _save_sub_pretrained(self, save_directory):
        self.data_config.save_pretrained(save_directory)
        self.feature_config.save_pretrained(save_directory)
        self.task_config.save_pretrained(save_directory)
        self.trainer_config.save_pretrained(save_directory)
        self.infer_config.save_pretrained(save_directory)

    def _post_dict(self, _dict):
        del _dict["data_config"]
        _dict["data_config_path"] = _config_type_name[self.data_config.config_type]

        del _dict["feature_config"]
        _dict["feature_config_path"] = _config_type_name[self.feature_config.config_type]

        del _dict["task_config"]
        _dict["task_config_path"] = _config_type_name[self.task_config.config_type]

        del _dict["trainer_config"]
        _dict["trainer_config_path"] = _config_type_name[self.trainer_config.config_type]

        del _dict["infer_config"]
        _dict["infer_config_path"] = _config_type_name[self.infer_config.config_type]
        return _dict

    @classmethod
    def _from_dict(cls, config_dict):
        data_config = BaseDataConfig.from_pretrained(
            os.path.join(config_dict["base_path"], config_dict["data_config_path"]))

        feature_config = BaseFeatureConfig.from_pretrained(
            os.path.join(config_dict["base_path"], config_dict["feature_config_path"]))

        task_config = BaseTaskConfig.from_pretrained(
            os.path.join(config_dict["base_path"], config_dict["task_config_path"]))

        trainer_config = BaseTrainerConfig.from_pretrained(
            os.path.join(config_dict["base_path"], config_dict["trainer_config_path"]))

        infer_config = BaseInferConfig.from_pretrained(
            os.path.join(config_dict["base_path"], config_dict["infer_config_path"]))

        return cls(data_config=data_config, feature_config=feature_config, task_config=task_config,
                   trainer_config=trainer_config, infer_config=infer_config, **config_dict)


