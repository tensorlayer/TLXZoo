import unittest
from tlxzoo.models.vgg.vgg import VGG
from tlxzoo.models.vgg.config_vgg import *
from tlxzoo.config.config import BaseImageFeatureConfig
from tlxzoo.models.vgg.feature_vgg import VGGFeature
import tensorlayerx as tlx
import numpy as np
import shutil


class ModelTestCase(unittest.TestCase):
    def setUp(self):
        print('setUp...')
        self.vgg16_model_config = VGGModelConfig(layer_type="vgg16")
        self.model = VGG(self.vgg16_model_config)
        self.model.set_eval()

    def tearDown(self):
        del self.model
        print('tearDown...')

    def test_config(self):
        self.vgg16_model_config.save_pretrained("vgg16")
        vgg16_model_config2 = VGGModelConfig.from_pretrained("vgg16")
        self.assertEqual(self.vgg16_model_config, vgg16_model_config2)
        shutil.rmtree("vgg16")

    def test_save_pretrained(self):
        self.model.save_pretrained("vgg16")
        model_2 = VGG.from_pretrained("vgg16")
        # self.assertEqual(self.model.all_weights[1], model_2.all_weights[1])
        assert all(self.model.all_weights[1] == model_2.all_weights[1])
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
        output = self.model(img)

        self.assertIsNotNone(output.output)


if __name__ == '__main__':
    unittest.main()

