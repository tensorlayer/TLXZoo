from ...config.config import BaseModelConfig, BaseTaskConfig, BaseImageFeatureConfig
from ...utils import MODEL_WEIGHT_NAME, TASK_WEIGHT_NAME
import os

cfg = {
    'A': [[64], 'M', [128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'B': [[64, 64], 'M', [128, 128], 'M', [256, 256], 'M', [512, 512], 'M', [512, 512], 'M', 'F', 'fc1', 'fc2', 'O'],
    'D':
        [
            [64, 64], 'M', [128, 128], 'M', [256, 256, 256], 'M', [512, 512, 512], 'M', [512, 512, 512], 'M', 'F',
            'fc1', 'fc2', 'O'
        ],
    'E':
        [
            [64, 64], 'M', [128, 128], 'M', [256, 256, 256, 256], 'M', [512, 512, 512, 512], 'M', [512, 512, 512, 512],
            'M', 'F', 'fc1', 'fc2', 'O'
        ],
}

mapped_cfg = {
    'vgg11': 'A',
    'vgg11_bn': 'A',
    'vgg13': 'B',
    'vgg13_bn': 'B',
    'vgg16': 'D',
    'vgg16_bn': 'D',
    'vgg19': 'E',
    'vgg19_bn': 'E'
}

model_urls = {
    'vgg16': 'http://www.cs.toronto.edu/~frossard/vgg16/',
    'vgg19': 'https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/'
}

model_saved_name = {'vgg16': 'vgg16_weights.npz', 'vgg19': 'vgg19.npy'}


class VGGModelConfig(BaseModelConfig):
    model_type = "vgg"

    def __init__(
            self,
            end_with='fc2_relu',
            fc2_units=4096,
            fc1_units=4096,
            layer_type="vgg16",
            batch_norm=False,
            layers=None,
            weights_path=MODEL_WEIGHT_NAME,
            **kwargs
    ):
        self.end_with = end_with
        self.batch_norm = batch_norm
        self.fc2_units = fc2_units
        self.fc1_units = fc1_units
        self.layer_type = layer_type
        if layers is None:
            self.layers = cfg[mapped_cfg[self.layer_type]]
        else:
            self.layers = layers
        if weights_path is None:
            self.weights_path = os.path.join(model_urls[self.layer_type], model_saved_name[self.layer_type])
        else:
            self.weights_path = weights_path

        super().__init__(
            **kwargs,
        )

    def _get_last_output_size(self):
        if self.end_with == "fc2_relu":
            return None, self.fc2_units
        elif self.end_with == "fc1_relu":
            return None, self.fc1_units
        else:
            raise ValueError(f"end_with must in ['fc1_relu', 'fc2_relu'], get {self.end_with}")


class VGGForImageClassificationTaskConfig(BaseTaskConfig):
    task_type = "vgg_for_image_classification"
    model_config_type = VGGModelConfig

    def __init__(self,
                 model_config: model_config_type,
                 num_labels=1000,
                 weights_path=TASK_WEIGHT_NAME,
                 **kwargs):
        self.num_labels = num_labels
        if weights_path is None:
            self.weights_path = model_config.weights_path
        else:
            self.weights_path = weights_path
        super(VGGForImageClassificationTaskConfig, self).__init__(model_config, **kwargs)
