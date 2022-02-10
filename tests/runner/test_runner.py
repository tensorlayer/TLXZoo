import unittest
from tlxzoo.config import BaseRunnerConfig
from tlxzoo.runner import Runner
from tlxzoo.dataset import DataLoaders, ImageClassificationDataConfig
from tlxzoo.models.vgg.task_vgg import VGGForImageClassification
from tlxzoo.models.vgg.config_vgg import *
from tlxzoo.config.config import BaseImageFeatureConfig
from tlxzoo.models.vgg.feature_vgg import VGGFeature


class ModelTestCase(unittest.TestCase):
    def test_run(self):
        # data
        config = ImageClassificationDataConfig()
        data_loaders = DataLoaders(config, train_limit=100)

        # task
        vgg16_model_config = VGGModelConfig()
        vgg16_task_config = VGGForImageClassificationTaskConfig(vgg16_model_config, num_labels=10)
        vgg16_task = VGGForImageClassification(vgg16_task_config)

        # feat
        image_feat_config = BaseImageFeatureConfig()
        vgg_feature = VGGFeature(image_feat_config)

        # run
        run_config = BaseRunnerConfig()
        run = Runner(model=vgg16_task, data_loader=data_loaders, config=run_config)
        run.register_feature_transform_hook(vgg_feature)
        n_epoch = 5
        print_freq = 2
        # train
        run.train(n_epoch=n_epoch, print_freq=print_freq, print_train_batch=False)


if __name__ == '__main__':
    unittest.main()

