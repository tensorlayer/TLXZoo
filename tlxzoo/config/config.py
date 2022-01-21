from tensorlayerx import logging
import abc


class PreTrainedMixin:
    """
    A few utilities for pretrain, to be used as a mixin.
    """


class BaseConfig(PreTrainedMixin):
    def __init__(self, **kwargs):
        self.pretrained_path = kwargs.pop("pretrained_path", ("", ""))

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logging.error(f"Can't set {key} with value {value} for {self}")
                raise err


class BaseModelConfig(BaseConfig, metaclass=abc.ABCMeta):
    config_type = "model"

    def get_last_output_size(self):
        size = self._get_last_output_size()
        if not isinstance(size, tuple):
            raise TypeError(f"Type of size is tuple, get {type(size)}")
        return size

    @abc.abstractmethod
    def _get_last_output_size(self):
        raise NotImplementedError


class BaseTaskConfig(BaseConfig):
    config_type = "task"

    def __init__(self, model_config: BaseModelConfig, **kwargs):
        self.model_config = model_config
        super(BaseTaskConfig, self).__init__(**kwargs)


class BaseInferConfig(BaseConfig):
    config_type = "infer"


class BaseDataConfig(BaseConfig):
    config_type = "data"


class BaseRunnerConfig(BaseConfig):
    config_type = "runner"


class BaseAppConfig(BaseConfig):
    config_type = "app"

