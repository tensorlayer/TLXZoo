import unittest
from tlxzoo.dataset import DataLoaders, ImageClassificationDataConfig


class DataTestCase(unittest.TestCase):
    def test_set_up(self):
        config = ImageClassificationDataConfig()
        data_loaders = DataLoaders(config)


if __name__ == '__main__':
    unittest.main()

