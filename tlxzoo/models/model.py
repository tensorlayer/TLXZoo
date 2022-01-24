from tensorlayerx import nn, logging
from tensorlayerx.files import assign_weights, maybe_download_and_extract
import os
import numpy as np


# class PreTrainedMixin:
#     """
#     A few utilities for pretrain, to be used as a mixin.
#     """
#
#     @classmethod
#     def from_pretrained(cls, pretrained_path, *model_args, **kwargs):
#         logging.info("Restore pre-trained parameters")
#         config = kwargs.pop("config", None)
#         from_tf = kwargs.pop("from_tf", False)
#         from_np = kwargs.pop("from_np", False)
#
#         if config is None:
#             pass
#
#         maybe_download_and_extract(
#             config.pretrained_path[0],
#             pretrained_model_name_or_path,
#             config.pretrained_path[1],
#         )
#         model_kwargs = kwargs
#
#         weights = []
#         model = cls(config, *model_args, **model_kwargs)
#
#         if from_tf:
#             try:
#                 import h5py
#             except Exception:
#                 raise ImportError('h5py not imported')
#
#             f = h5py.File(os.path.join(pretrained_model_name_or_path, config.pretrained_path[0]), 'r')
#
#         if from_np:
#             npz = np.load(os.path.join('model', config.pretrained_path[0]), allow_pickle=True)
#             # get weight list
#             for val in sorted(npz.items()):
#                 logging.info("  Loading weights %s in %s" % (str(val[1].shape), val[0]))
#                 weights.append(val[1])
#                 if len(model.all_weights) == len(weights):
#                     break
#
#         # assign weight values
#         assign_weights(weights, model)
#         del weights


class BaseModule(nn.Module):
    def __init__(self, config, *args, **kwargs):
        """Initialize BaseModule, inherited from `tensorlayerx.nn.Module`"""

        super(BaseModule, self).__init__(*args, **kwargs)
        self.config = config

    @classmethod
    def from_pretrained(cls, pretrained_base_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)

        if config is None:
            if pretrained_base_path is None:
                raise ValueError("pretrained_base_path and config are both None.")
            config = cls.config_class.from_pretrained(pretrained_base_path, **kwargs)

        model = cls(config, *model_args, **kwargs)

        if pretrained_base_path is None:
            logging.warning("Don't load weight.")
            return model

        weights_path = config.weights_path
        model.load_weights(weights_path)
        return model
