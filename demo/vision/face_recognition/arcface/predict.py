from tlxzoo.module.arcface import *

size = 112
arc_model = ArcFace(size=size)
arc_model.load_weights("demo/vision/face_recognition/arcface/model.npz")
arc_model.set_eval()

import cv2
import numpy as np
img_path = "temp.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (size, size))
img = img.astype(np.float32) / 255.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
if len(img.shape) == 3:
    img = np.expand_dims(img, 0)

img = tlx.convert_to_tensor(img)

embeds = l2_norm(arc_model(img))

