import tensorlayerx as tlx
from tlxzoo.vision.human_pose_estimation import HumanPoseEstimation, inference
from tlxzoo.module.hrnet import HRNetTransform
from PIL import Image
import numpy as np


if __name__ == '__main__':
    transform = HRNetTransform()

    model = HumanPoseEstimation("hrnet")
    model.load_weights("./demo/vision/human_pose_estimation/hrnet/model.npz")
    model.set_eval()

    path = "./demo/vision/human_pose_estimation/hrnet/hrnet.jpg"
    image = Image.open(path).convert('RGB')
    image_height, image_width = image.height, image.width
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = transform.resize(image, (transform.size[0], transform.size[1]))

    image = tlx.convert_to_tensor([image])

    inference(image_tensor=image, model=model, image_dir=path, original_image_size=[image_height, image_width])


