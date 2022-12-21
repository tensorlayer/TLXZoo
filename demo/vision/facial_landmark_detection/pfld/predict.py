import tensorlayerx as tlx
from tensorlayerx.vision.transforms import *
from tensorlayerx.vision.utils import load_image, save_image
from tlxzoo.vision.facial_landmark_detection import FacialLandmarkDetection, draw_landmarks


if __name__ == '__main__':
    model = FacialLandmarkDetection(backbone="pfld")
    model.load_weights("./demo/vision/facial_landmark_detection/pfld/model.npz")
    model.set_eval()

    transform = Compose([
        Resize((112, 112)),
        Normalize(mean=[0, 0, 0], std=[255.0, 255.0, 255.0]),
        ToTensor()
    ])
    image = load_image("./demo/vision/facial_landmark_detection/pfld/face.jpg")
    input = tlx.expand_dims(transform(image), 0)

    landmarks, _ = model.predict(input)
    landmarks = tlx.convert_to_numpy(landmarks[0]).reshape((-1, 2))
    image = draw_landmarks(image, landmarks)
    save_image(image, 'result.jpg', './demo/vision/facial_landmark_detection/pfld/')
    
