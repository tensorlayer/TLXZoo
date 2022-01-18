from tensorlayerx import nn, logging
from tensorlayerx.files import maybe_download_and_extract
import os


class PreTrainedMixin:
    """
    A few utilities for pretrain, to be used as a mixin.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        logging.info("Restore pre-trained parameters")
        config = kwargs.pop("config", None)
        from_tf = kwargs.pop("from_tf", False)

        if config is None:
            # TODO: cls.config_class.from_pretrained
            pass

        maybe_download_and_extract(
            config.pretrained_path[0],
            pretrained_model_name_or_path,
            config.pretrained_path[1],
        )

        if from_tf:
            try:
                import h5py
            except Exception:
                raise ImportError('h5py not imported')

            f = h5py.File(os.path.join(pretrained_model_name_or_path, 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'),
                          'r')


class BaseModule(nn.Module, PreTrainedMixin):
    def __init__(self, config):
        """Initialize BaseModule, inherited from `tensorlayerx.nn.Module`"""

        super(BaseModule, self).__init__()
        self.config = config
