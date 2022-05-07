from tlxzoo.datasets import DataLoaders
from tlxzoo.module.detr import DetrTransform, post_process
from tlxzoo.vision.object_detection import ObjectDetection
from tlxzoo.datasets.coco import CocoEvaluator
import tensorlayerx as tlx
from PIL import Image


if __name__ == '__main__':
    transform = DetrTransform()

    model = ObjectDetection(backbone="detr")
    model.load_weights("demo/vision/object_detection/detr/model.npz")
    model.set_eval()

    image_path = "./000000039769.jpeg"
    image = Image.open(image_path).convert('RGB')
    orig_size = image.size

    image, _ = transform(image, None)

    inputs = tlx.convert_to_tensor([image])

    outputs = model(inputs=inputs)

    orig_target_sizes = tlx.convert_to_tensor([orig_size], dtype=tlx.float32)
    results = post_process(outputs["pred_logits"], outputs["pred_boxes"], orig_target_sizes)

    for i in results:
        for s, l, b in zip(i["scores"], i["labels"], i["boxes"]):
            if s <= 0.5:
                continue
            print(s, l, b)

