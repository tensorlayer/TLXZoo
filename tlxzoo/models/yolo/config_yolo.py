from ...config.config import BaseConfig

weights_url = {'link': 'https://pan.baidu.com/s/1MC1dmEwpxsdgHO1MZ8fYRQ', 'password': 'idsz'}


class YOLOv4Config(BaseConfig):
    model_type = "yolov4"

    def __init__(
            self,
            num_class=80,
            **kwargs
    ):
        self.num_class = num_class

        pretrained_path = ("", "")

        super().__init__(
            pretrained_path=pretrained_path,
            **kwargs,
        )