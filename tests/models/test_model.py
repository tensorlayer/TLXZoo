import unittest
from tlxzoo.models.vgg.vgg import VGG
from tlxzoo.models.vgg.config_vgg import *
import tensorlayerx as tlx
import numpy as np


class ModelTestCase(unittest.TestCase):
    def setUp(self):
        print('setUp...')
        vgg16_model_config = VGG16ModelConfig()
        self.model = VGG(vgg16_model_config)
        self.model.set_eval()

    def tearDown(self):
        del self.model
        print('tearDown...')

    def test_from_pretrained(self):
        ...

    def test_save_pretrained(self):
        ...

    def test_feature(self):
        ...

    def test_call(self):
        img = tlx.vis.read_image('../elephant.jpeg')
        img = tlx.prepro.imresize(img, (224, 224)).astype(np.float32) / 255
        output = self.model(img)

        self.assertIsNotNone(output.output)


if __name__ == '__main__':
    unittest.main()

