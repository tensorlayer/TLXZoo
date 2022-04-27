from tlxzoo.dataset import DataLoaders, FaceRecognitionDataConfig
from tlxzoo.models.retinaface import *
from tlxzoo.models.retinaface.task_retinaface import load_ckpt_weight
import os

feat_config = RetinaFaceFeatureConfig()
feat = RetinaFaceFeature(feat_config)

model_config = RetinaFaceModelConfig()
task_config = RetinaFaceForFaceRecognitionTaskConfig(model_config)

retina_face_model = RetinaFaceForFaceRecognition(task_config)
retina_face_model = load_ckpt_weight(retina_face_model, "./checkpoints/retinaface_res50")

retina_face_model.set_eval()
feat.set_eval()

from tlxzoo.models.arcface import *

size = 112
arc_model = ArcFace(size=size)
load_ckpt_weight(arc_model, "./checkpoints/arc_res50")
arc_model.set_eval()

img_file = "./face_recognition.png"
img, (labels, pad_params, image_path) = feat(img_file, None)

img = tlx.convert_to_tensor([img])

output = retina_face_model(img)
outputs = feat.decode_one(output[0], output[1], output[2], img, pad_params, score_th=0.5)

# 存储
img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
# img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
img_height_raw, img_width_raw, _ = img_raw.shape
for prior_index in range(len(outputs)):
    draw_bbox_landm(img_raw, outputs[prior_index], img_height_raw, img_width_raw, prior_index)
cv2.imwrite("./temp.jpg", img_raw)

embs = []
img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
for index, ann in enumerate(outputs):
    x1, y1, x2, y2 = int(ann[0] * img_width_raw), int(ann[1] * img_height_raw), \
                     int(ann[2] * img_width_raw), int(ann[3] * img_height_raw)
    print(x1, y1, x2, y2)

    face = img_raw[y1:y2, x1:x2]
    img = cv2.resize(face, (size, size))
    img = img.astype(np.float32) / 255.

    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    emb = l2_norm(arc_model(img))

    embs.append(tlx.convert_to_numpy(emb)[0])

from sklearn.metrics.pairwise import cosine_similarity

s = cosine_similarity(embs)
print(s)


