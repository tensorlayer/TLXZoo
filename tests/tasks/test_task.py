import unittest
from tlxzoo.models.vgg import *
from tlxzoo.config import BaseImageFeatureConfig
import tensorlayerx as tlx
import shutil


class TaskTestCase(unittest.TestCase):
    def setUp(self):
        print('setUp...')
        self.vgg16_task_config = VGGForImageClassificationTaskConfig()
        self.task = VGGForImageClassification(self.vgg16_task_config)
        self.task.set_eval()

    def tearDown(self):
        del self.task
        print('tearDown...')

    def test_config(self):
        self.vgg16_task_config.save_pretrained("vgg16")
        vgg16_task_config2 = VGGForImageClassificationTaskConfig.from_pretrained("vgg16")
        self.assertEqual(self.vgg16_task_config, vgg16_task_config2)
        shutil.rmtree("vgg16")

    def test_save_pretrained(self):
        self.task.save_pretrained("vgg16")
        task_2 = VGGForImageClassification.from_pretrained("vgg16")
        assert all(self.task.all_weights[1] == task_2.all_weights[1])
        shutil.rmtree("vgg16")

    def test_feature(self):
        image_feat_config = BaseImageFeatureConfig()
        image_feat_config.save_pretrained("vgg16")
        image_feat_config2 = BaseImageFeatureConfig.from_pretrained("vgg16")
        shutil.rmtree("vgg16")
        self.assertEqual(image_feat_config, image_feat_config2)

        vgg_feature = VGGFeature(image_feat_config)
        vgg_feature.save_pretrained("vgg16")
        vgg_feature2 = VGGFeature.from_pretrained("vgg16")
        shutil.rmtree("vgg16")
        self.assertEqual(vgg_feature.config, vgg_feature2.config)

    def test_call(self):
        image_feat_config = BaseImageFeatureConfig()
        vgg_feature = VGGFeature(image_feat_config)
        img = tlx.vis.read_image('../elephant.jpeg')

        img = vgg_feature([img])
        output = self.task(img, return_output=True)

        self.assertIsNotNone(output.logits)