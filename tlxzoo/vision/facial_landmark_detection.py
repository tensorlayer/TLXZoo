import cv2
import tensorlayerx as tlx
from tlxzoo.module.pfld import PFLD


class FacialLandmarkDetection(tlx.nn.Module):
    def __init__(self, backbone, **kwargs):
        super(FacialLandmarkDetection, self).__init__()
        if backbone == 'pfld':
            self.backbone = PFLD(**kwargs)
        else:
            raise ValueError(f"tlxzoo don`t support {backbone}")

    def loss_fn(self, output, target, name="", **kwargs):
        if hasattr(self.backbone, "loss_fn"):
            return self.backbone.loss_fn(output, target)
        else:
            raise ValueError("loss fn isn't defined.")

    def forward(self, inputs, **kwargs):
        return self.backbone(inputs, **kwargs)

    def predict(self, inputs):
        self.set_eval()
        return self.backbone(inputs)


def draw_landmarks(image, landmarks, radius=1, color=(0, 0, 255), normalized=True):
    image = image.copy()
    if normalized:
        height, width = image.shape[:2]
        landmarks = landmarks.copy()
        landmarks[:, 0] *= width
        landmarks[:, 1] *= height
    for x, y in landmarks:
        cv2.circle(image, (int(x), int(y)), radius, color)
    return image