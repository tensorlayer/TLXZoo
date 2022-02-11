import unittest
import shutil

from tlxzoo.config import *
from tlxzoo.dataset import ImageClassificationDataConfig
from tlxzoo.models.vgg.config_vgg import *
from tlxzoo.config.config import BaseImageFeatureConfig


class DataTestCase(unittest.TestCase):
    def test_set_up(self):
        data_config = ImageClassificationDataConfig(per_device_train_batch_size=8)

        vgg16_model_config = VGGModelConfig()
        vgg16_task_config = VGGForImageClassificationTaskConfig(vgg16_model_config, num_labels=10)

        image_feat_config = BaseImageFeatureConfig()

        run_config = BaseRunnerConfig()

        infer_config = BaseInferConfig()

        app_config = BaseAppConfig(data_config=data_config, feature_config=image_feat_config,
                                   task_config=vgg16_task_config, runner_config=run_config, infer_config=infer_config)

        app_config.save_pretrained("vgg")

        app_config = BaseAppConfig.from_pretrained("vgg")
        shutil.rmtree("vgg")