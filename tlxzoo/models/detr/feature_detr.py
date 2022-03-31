from ...feature.feature import BaseImageFeature
from ...config.config import BaseImageFeatureConfig
import numpy as np
from ...utils.registry import Registers
import tensorlayerx as tlx
import PIL


@Registers.feature_configs.register
class DetrFeatureConfig(BaseImageFeatureConfig):
    def __init__(self,
                 do_resize=True,
                 do_normalize=True,
                 size=800,
                 max_size=1333,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 **kwargs):
        self.size = size
        self.max_size = max_size

        super(DetrFeatureConfig, self).__init__(do_resize=do_resize,
                                                do_normalize=do_normalize, mean=mean, std=std, **kwargs)


@Registers.features.register
class DetrFeature(BaseImageFeature):
    config_class = DetrFeatureConfig

    def __init__(
            self,
            config,
            **kwargs
    ):
        self.config = config

        super(DetrFeature, self).__init__(config, **kwargs)

    def prepare_coco_detection(self, image, anno, return_segmentation_masks=True):
        """
        Convert the target in COCO format into the format expected by DETR.
        """
        w, h = image.size

        # get all COCO annotations for the given image

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = boxes[:, 0::2].clip(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = np.asarray(classes, dtype=np.int64)

        if return_segmentation_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = np.asarray(keypoints, dtype=np.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.reshape((-1, 3))

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if return_segmentation_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["class_labels"] = classes
        if return_segmentation_masks:
            target["masks"] = masks
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = np.asarray([obj["area"] for obj in anno], dtype=np.float32)
        iscrowd = np.asarray([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno], dtype=np.int64)
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = np.asarray([int(h), int(w)], dtype=np.int64)
        target["size"] = np.asarray([int(h), int(w)], dtype=np.int64)

        return image, target

    def convert_coco_poly_to_mask(self, segmentations, height, width):

        try:
            from pycocotools import mask as coco_mask
        except ImportError:
            raise ImportError("Pycocotools is not installed in your environment.")

        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = np.asarray(mask, dtype=np.uint8)
            mask = np.any(mask, axis=2)
            masks.append(mask)
        if masks:
            masks = np.stack(masks, axis=0)
        else:
            masks = np.zeros((0, height, width), dtype=np.uint8)

        return masks

    def resize(self, image, size, resample=PIL.Image.BILINEAR):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, list):
            size = tuple(size)
        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        return image.resize(size, resample=resample)

    def _resize(self, image, size, target=None, max_size=None):

        def get_size_with_aspect_ratio(image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

        def get_size(image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size
            else:
                # size returned must be (w, h) since we use PIL to resize images
                # so we revert the tuple
                return get_size_with_aspect_ratio(image_size, size, max_size)[::-1]

        size = get_size(image.size, size, max_size)
        rescaled_image = self.resize(image, size=size)

        if target is None:
            return rescaled_image, None

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * np.asarray([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        w, h = size
        target["size"] = np.asarray([h, w], dtype=np.int64)

        if "masks" in target:
            # masks = tlx.convert_to_tensor(target["masks"][:, None])
            # masks = tlx.cast(masks, dtype=tlx.float32)
            # interpolated_masks = tlx.vision.transforms.resize(masks, (h, w), method="nearest")[:, 0] > 0.5
            # target["masks"] = tlx.convert_to_numpy(interpolated_masks)
            masks = np.transpose(target["masks"], axes=[1, 2, 0]).astype(float)
            interpolated_masks = tlx.vision.transforms.resize(masks, (h, w), method="nearest") > 0.5
            interpolated_masks = np.transpose(interpolated_masks, axes=[2, 0, 1])
            target["masks"] = interpolated_masks

        return rescaled_image, target

    def to_numpy_array(self, image, rescale=None, channel_first=False):

        if isinstance(image, PIL.Image.Image):
            image = np.array(image)

        if rescale is None:
            rescale = isinstance(image.flat[0], np.integer)

        if rescale:
            image = image.astype(np.float32) / 255.0

        if channel_first and image.ndim == 3:
            image = image.transpose(2, 0, 1)

        return image

    def normalize(self, image, mean, std):

        if isinstance(image, PIL.Image.Image):
            image = self.to_numpy_array(image)

        if isinstance(image, np.ndarray):
            if not isinstance(mean, np.ndarray):
                mean = np.array(mean).astype(image.dtype)
            if not isinstance(std, np.ndarray):
                std = np.array(std).astype(image.dtype)

        if image.ndim == 3 and image.shape[0] in [1, 3]:
            return (image - mean[:, None, None]) / std[:, None, None]
        else:
            return (image - mean) / std

    def _normalize(self, image, mean, std, target=None):
        """
        Normalize the image with a certain mean and std.

        If given, also normalize the target bounding boxes based on the size of the image.
        """

        image = self.normalize(image, mean=mean, std=std)
        if target is None:
            return image, None

        target = target.copy()
        h, w = image.shape[:2]

        if "boxes" in target:
            boxes = target["boxes"]
            boxes = corners_to_center_format(boxes)
            boxes = boxes / np.asarray([w, h, w, h], dtype=np.float32)
            target["boxes"] = boxes

        return image, target

    def prepare_coco_panoptic(self, image, target, masks_path, return_masks=True):
        w, h = image.size
        ann_info = target.copy()
        ann_path = pathlib.Path(masks_path) / ann_info["file_name"]

        if "segments_info" in ann_info:
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb_to_id(masks)

            ids = np.array([ann["id"] for ann in ann_info["segments_info"]])
            masks = masks == ids[:, None, None]
            masks = np.asarray(masks, dtype=np.uint8)

            labels = np.asarray([ann["category_id"] for ann in ann_info["segments_info"]], dtype=np.int64)

        target = {}
        target["image_id"] = np.asarray(
            [ann_info["image_id"] if "image_id" in ann_info else ann_info["id"]], dtype=np.int64
        )
        if return_masks:
            target["masks"] = masks
        target["class_labels"] = labels

        target["boxes"] = masks_to_boxes(masks)

        target["size"] = np.asarray([int(h), int(w)], dtype=np.int64)
        target["orig_size"] = np.asarray([int(h), int(w)], dtype=np.int64)
        if "segments_info" in ann_info:
            target["iscrowd"] = np.asarray([ann["iscrowd"] for ann in ann_info["segments_info"]], dtype=np.int64)
            target["area"] = np.asarray([ann["area"] for ann in ann_info["segments_info"]], dtype=np.float32)

        return image, target

    def collate_fn(self, data):
        images = [i[0] for i in data]
        padded_images, pixel_mask = self.pad_and_create_pixel_mask(images)
        new_data = []
        labels = []
        for i, j, k in zip(data, padded_images, pixel_mask):
            labels.append(i[1])
            new_data.append({"inputs": j, "pixel_mask": k})
        if len(data) >= 2:
            return tlx.dataflow.dataloader.utils.default_collate(new_data), labels
        else:
            data = {}
            for i, j in new_data[0].items():
                data[i] = np.array([j])
            # label = {}
            # for i, j in labels[0].items():
            #     label[i] = np.array([j])
            return tlx.dataflow.dataloader.utils.default_convert(data), labels

    def __call__(self, image, label, *args, **kwargs):
        if label is not None:
            image_id = label["image_id"]
            label = label["annotations"]
            image, target = self.prepare_coco_detection(image, label)
        else:
            image_id = None
            target = None
        if self.config.do_resize and self.config.size is not None:
            image, target = self._resize(image=image, target=target, size=self.config.size,
                                         max_size=self.config.max_size)

        if self.config.do_normalize:
            image, target = self._normalize(
                image=image, mean=self.config.mean, std=self.config.std, target=target
            )

        if image_id:
            target["image_id"] = image_id

        return image, target

    def _max_by_axis(self, the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def pad_and_create_pixel_mask(
        self, pixel_values_list
    ):

        max_size = self._max_by_axis([list(image.shape) for image in pixel_values_list])
        h, w, c = max_size
        padded_images = []
        pixel_mask = []
        for image in pixel_values_list:
            # create padded image
            padded_image = np.zeros((h, w, c), dtype=np.float32)
            padded_image[: image.shape[0], : image.shape[1], : image.shape[2]] = np.copy(image)
            padded_images.append(padded_image)
            # create pixel mask
            mask = np.zeros((h, w), dtype=bool)
            mask[: image.shape[0], : image.shape[1]] = True
            pixel_mask.append(mask)

        return padded_images, pixel_mask


def corners_to_center_format(x):
    """
    Converts a NumPy array of bounding boxes of shape (number of bounding boxes, 4) of corners format (x_0, y_0, x_1,
    y_1) to center format (center_x, center_y, width, height).
    """
    x_transposed = x.T
    x0, y0, x1, y1 = x_transposed[0], x_transposed[1], x_transposed[2], x_transposed[3]
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return np.stack(b, axis=-1)


def post_process(out_logits, out_bbox, target_sizes):
    prob = tlx.softmax(out_logits, -1)
    scores = tlx.reduce_max(prob[..., :-1], axis=-1)
    labels = tlx.argmax(prob[..., :-1], axis=-1)

    # convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(out_bbox)
    # and from relative [0, 1] to absolute [0, height] coordinates
    # img_h, img_w = target_sizes.unbind(1)
    img_h = target_sizes[:, 0]
    img_w = target_sizes[:, 1]
    scale_fct = tlx.stack([img_w, img_h, img_w, img_h], axis=1)
    boxes = boxes * scale_fct[:, None, :]

    results = []
    for s, l, b in zip(scores, labels, boxes):
        # indices = tlx.where(l != 91, None, None)
        # indices = tlx.squeeze(indices, axis=-1)
        #
        # s = tlx.gather(s, indices)
        # l = tlx.gather(l, indices)
        # b = tlx.gather(b, indices)

        indices = tlx.where(l != 0, None, None)
        indices = tlx.squeeze(indices, axis=-1)

        s = tlx.gather(s, indices)
        l = tlx.gather(l, indices)
        b = tlx.gather(b, indices)

        # indices = tlx.where(s >= 0.5, None, None)
        # indices = tlx.squeeze(indices, axis=-1)
        #
        # s = tlx.gather(s, indices)
        # l = tlx.gather(l, indices)
        # b = tlx.gather(b, indices)

        results.append({"scores": s, "labels": l, "boxes": b})

    return results


def post_process_segmentation(outputs, target_sizes, threshold=0.9, mask_threshold=0.5):

    out_logits, raw_masks = outputs["pred_logits"], outputs["pred_masks"]
    preds = []

    for cur_logits, cur_masks, size in zip(out_logits, raw_masks, target_sizes):
        # we filter empty queries and detection below threshold
        cur_logits = tlx.softmax(cur_logits, axis=-1)
        scores = tlx.reduce_max(cur_logits, axis=-1)
        labels = tlx.argmax(cur_logits, axis=-1)

        keep = tlx.not_equal(labels, (tlx.get_tensor_shape(outputs["pred_logits"])[-1] - 1)) & (scores > threshold)

        cur_scores = tlx.reduce_max(cur_logits, axis=-1)
        cur_classes = tlx.argmax(cur_logits, axis=-1)

        keep = tlx.convert_to_numpy(keep)
        cur_scores = tlx.convert_to_numpy(cur_scores)
        cur_classes = tlx.convert_to_numpy(cur_classes)
        cur_masks = tlx.convert_to_numpy(cur_masks)
        cur_scores = cur_scores[keep]
        cur_classes = cur_classes[keep]
        cur_masks = cur_masks[keep]

        cur_masks = np.transpose(cur_masks, axes=[1, 2, 0])

        cur_masks = tlx.vision.transforms.resize(cur_masks, tuple(size), method="bilinear")
        cur_masks = np.transpose(cur_masks, axes=[2, 0, 1])

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        cur_masks = (sigmoid(cur_masks) > mask_threshold) * 1

        predictions = {"scores": cur_scores, "labels": cur_classes, "masks": cur_masks}
        preds.append(predictions)
    return preds


def center_to_corners_format(x):
    # x_c, y_c, w, h = x.unbind(-1)
    x_c = x[..., 0]
    y_c = x[..., 1]
    w = x[..., 2]
    h = x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return tlx.stack(b, axis=-1)