import tensorlayerx as tlx
from tensorlayerx.vision.transforms import *
from tensorlayerx.vision.utils import load_image, save_image
from tlxzoo.vision.human_pose_estimation import HumanPoseEstimation, inference

if __name__ == '__main__':
    model = HumanPoseEstimation("hrnet")
    model.load_weights("./demo/vision/human_pose_estimation/hrnet/model.npz")
    model.set_eval()

    path = "./demo/vision/human_pose_estimation/hrnet/hrnet.jpg"
    image = load_image(path)
    image_height, image_width = image.shape[:2]
    
    transform = Compose([
        Resize((256, 256)),
        Normalize(mean=(0, 0, 0), std=(255.0, 255.0, 255.0)),
        ToTensor()
    ])
    image_tensor = transform(image)
    image_tensor = tlx.expand_dims(image_tensor, 0)

    image = inference(image_tensor=image_tensor, model=model, image=image, original_image_size=[image_height, image_width])
    save_image(image, 'result.jpg', './demo/vision/human_pose_estimation/hrnet')
