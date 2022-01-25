from tlxzoo.models.vgg import *
from tlxzoo.config import BaseImageFeatureConfig
import tensorlayerx as tlx

vgg16_task_config = VGGForImageClassificationTaskConfig()
vgg16_task = VGGForImageClassification(vgg16_task_config)
vgg16_task.set_eval()

image_feat_config = BaseImageFeatureConfig()
vgg_feature = VGGFeature(image_feat_config)

img = tlx.vis.read_image('./elephant.jpeg')
imgs = vgg_feature([img])
output = vgg16_task(imgs)
print(output.logits.shape)
