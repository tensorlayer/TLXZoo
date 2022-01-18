from tensorlayerx import logging


class PreTrainedMixin:
    """
    A few utilities for pretrain, to be used as a mixin.
    """


class BaseConfig(PreTrainedMixin):
    def __init__(self, **kwargs):
        self.pretrained_path = kwargs.pop("pretrained_path", ("", ""))

        self._kwargs = {}

        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logging.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def __setattr__(self, key, value):
        self._kwargs[key] = value

    def __getattr__(self, key: str):
        if key not in self._kwargs:
            logging.error(f"can't get {key} for {self}")
