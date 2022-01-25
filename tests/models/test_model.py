import unittest
from tlxzoo.models.vgg.vgg import VGG
from tlxzoo.models.vgg.config_vgg import *
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
        img = tlx.vis.read_image('../elephant.jpeg')
        img = tlx.prepro.imresize(img, (224, 224)).astype(np.float32) / 255

        output = self.model(img)
        self.assertIsNotNone(output.output)

    def test_call(self):
        img = tlx.vis.read_image('../elephant.jpeg')
        img = tlx.prepro.imresize(img, (224, 224)).astype(np.float32) / 255
        output = self.model(img)

        self.assertIsNotNone(output.output)


if __name__ == '__main__':
    unittest.main()

