from ...config.config import BaseModelConfig, BaseTaskConfig

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
    model_type = ""

    def __init__(
            self,
            end_with='fc2_relu',
            fc2_units=4096,
            fc1_units=4096,
            batch_norm=False,
            layers=None,
            **kwargs
    ):
        self.end_with = end_with
        self.batch_norm = batch_norm
        self.fc2_units = fc2_units
        self.fc1_units = fc1_units
        if layers is None:
            self.layers = cfg[mapped_cfg[self.model_type]]
        else:
            self.layers = layers

        pretrained_path = (model_saved_name[self.model_type], model_urls[self.model_type])

        super().__init__(
            pretrained_path=pretrained_path,
            **kwargs,
        )

    def _get_last_output_size(self):
        if self.end_with == "fc2_relu":
            return None, self.fc2_units
        elif self.end_with == "fc1_relu":
            return None, self.fc1_units
        else:
            raise ValueError(f"end_with must in ['fc1_relu', 'fc2_relu'], get {self.end_with}")


class VGG16ModelConfig(VGGModelConfig):
    model_type = "vgg16"


class VGG19ModelConfig(VGGModelConfig):
    model_type = "vgg19"


class VGGForImageClassificationTaskConfig(BaseTaskConfig):
    task_type = "vgg_for_image_classification"

    def __init__(self, model_config: BaseModelConfig, num_labels=1000, **kwargs):
        self.num_labels = num_labels
        super(VGGForImageClassificationTaskConfig, self).__init__(model_config, **kwargs)
