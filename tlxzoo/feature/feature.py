

class BaseFeature(object):
    ...

class FeaturePreTrainedMixin:
    @classmethod
    def from_pretrained(
            cls, pretrained_path, **kwargs
    ):
        config = cls.config_class.from_pretrained(pretrained_path, **kwargs)

        return cls(config)

    def save_pretrained(self, save_path):
        self.config.save_pretrained(save_path)
