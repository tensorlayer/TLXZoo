import unittest
import shutil

from tlxzoo.config import *
from tlxzoo.dataset import ImageClassificationDataConfig
from tlxzoo.models.vgg.config_vgg import *
from tlxzoo.config.config import BaseImageFeatureConfig


class RunnerTestCase(unittest.TestCase):
    def test_set_up(self):
        data_config = ImageClassificationDataConfig(per_device_train_batch_size=8)

        vgg16_model_config = VGGModelConfig()
        vgg16_task_config = VGGForImageClassificationTaskConfig(vgg16_model_config, num_labels=10)

        image_feat_config = BaseImageFeatureConfig()

        trainer_config = BaseTrainerConfig()

        infer_config = BaseInferConfig()

        runner_config = BaseRunnerConfig(data_config=data_config, feature_config=image_feat_config,
                                         task_config=vgg16_task_config, trainer_config=trainer_config,
                                         infer_config=infer_config)

        runner_config.save_pretrained("vgg")

        runner_config = BaseRunnerConfig.from_pretrained("vgg")
        shutil.rmtree("vgg")
