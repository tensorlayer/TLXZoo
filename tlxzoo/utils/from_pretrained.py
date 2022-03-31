from tensorlayerx import nn, logging
import os
from tensorlayerx.files import maybe_download_and_extract
import tensorlayerx as tlx


class ModuleFromPretrainedMixin:
    @classmethod
    def from_pretrained(cls, pretrained_base_path, *args, **kwargs):
        config = kwargs.pop("config", None)

        if config is None:
            if pretrained_base_path is None:
                raise ValueError("pretrained_base_path and config are both None.")
            config = cls.config_from_pretrained(pretrained_base_path, **kwargs)

        module = cls.from_config(config, *args, **kwargs)

        load_weight = kwargs.pop("load_weight", True)
        if not load_weight:
            return module

        weights_path = config.weights_path

        weight_from = kwargs.pop("weight_from", "tlx")
        if weight_from == "huggingface":
            if tlx.ops.load_backend.BACKEND == "tensorflow":
                return module._load_huggingface_tf_weight(pretrained_base_path)
            elif tlx.ops.load_backend.BACKEND == "pytorch":
                return module._load_huggingface_pt_weight(pretrained_base_path)
            else:
                raise ValueError(f"BACKEND: {tlx.ops.load_backend.BACKEND} do not support huggingface weight loading.")

        if not weights_path.startswith("http"):
            if pretrained_base_path is None:
                logging.warning("Don't load weight.")
                return module

            weights_path = os.path.join(pretrained_base_path, weights_path)
            if not os.path.exists(weights_path):
                raise ValueError(f"{weights_path} is not exist.")
        else:
            if pretrained_base_path is None:
                pretrained_base_path = ".cache"
            url, name = weights_path.rsplit("/", 1)
            maybe_download_and_extract(name, pretrained_base_path, url + "/")
            weights_path = os.path.join(pretrained_base_path, name)

        format = kwargs.pop("format", None)
        module.load_weights(weights_path, format=format)
        return module

    def _load_huggingface_tf_weight(self, weight_path):
        raise NotImplementedError

    def _load_huggingface_pt_weight(self, weight_path):
        raise NotImplementedError

    def _save_pretrained(self, save_directory, name, format, save_weight=True):
        if os.path.isfile(save_directory):
            logging.error(f"Save directory ({save_directory}) should be a directory, not a file.")
            return

        os.makedirs(save_directory, exist_ok=True)

        # save config
        self.config.save_pretrained(save_directory)

        if save_weight:
            # save weight
            self.save_weights(os.path.join(save_directory, name), format)