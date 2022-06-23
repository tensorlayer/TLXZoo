import cv2
import numpy as np
import tensorlayerx as tlx
from tlxzoo.vision.facial_landmark_detection import FacialLandmarkDetection


if __name__ == '__main__':
    model = FacialLandmarkDetection(backbone="pfld")
    model.load_weights("./model.npz")
    model.set_eval()

    image = cv2.imread("face.jpg")
    image = cv2.resize(image, (112, 112))
    image = image.astype(np.float32) / 255.0
    image = tlx.convert_to_tensor([image])

    landmarks, _ = model.predict(image)
    print(landmarks)
