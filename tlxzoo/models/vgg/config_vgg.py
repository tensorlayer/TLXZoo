from ...config.config import BaseConfig

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


class VGGConfig(BaseConfig):
    model_type = ""

    def __init__(
            self,
            end_with='outputs',
            batch_norm=False,
            layers=None,
            **kwargs
    ):
        self.end_with = end_with
        self.batch_norm = batch_norm
        if layers is None:
            self.layers = cfg[mapped_cfg[self.model_type]]
        else:
            self.layers = layers

        pretrained_path = (model_saved_name[self.model_type], model_urls[self.model_type])

        super().__init__(
            pretrained_path=pretrained_path,
            **kwargs,
        )


class VGG16Config(VGGConfig):
    model_type = "vgg16"


class VGG19Config(VGGConfig):
    model_type = "vgg19"
