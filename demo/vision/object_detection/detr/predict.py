import tensorlayerx as tlx
from PIL import Image
from tensorlayerx.vision.transforms import Compose

from tlxzoo.module.detr import *
from tlxzoo.vision.object_detection import ObjectDetection


if __name__ == '__main__':
    model = ObjectDetection(backbone="detr")
    model.load_weights("demo/vision/object_detection/detr/model.npz")
    model.set_eval()

    image_path = "demo/vision/object_detection/detr/000000039769.jpeg"
    image = Image.open(image_path).convert('RGB')
    orig_size = image.size

    transform = Compose([
        Resize(size=800, max_size=1333),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image, _ = transform((image, None))

    inputs = tlx.convert_to_tensor([image])

    outputs = model(inputs=inputs)

    orig_target_sizes = tlx.convert_to_tensor([orig_size], dtype=tlx.float32)
    results = post_process(outputs["pred_logits"], outputs["pred_boxes"], orig_target_sizes)

    for i in results:
        for s, l, b in zip(i["scores"], i["labels"], i["boxes"]):
            if s <= 0.5:
                continue
            print(s, l, b)

