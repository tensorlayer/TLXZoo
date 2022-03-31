import numpy as np
import cv2
from ...utils.registry import Registers
from ..dataset import BaseDataSetDict, Dataset, BaseDataSetMixin
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorlayerx as tlx
import pycocotools.mask as mask_util
import contextlib
import os
from PIL import Image
import copy


@Registers.datasets.register("Coco")
class CocoDetection(Dataset, BaseDataSetMixin):
    def __init__(
        self, root, annFile, transforms=None, limit=None,
    ):
        self.coco = COCO(annFile)
        self.root = root
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []
        self.ids = list(sorted(self.coco.imgs.keys()))
        if limit:
            self.ids = self.ids[:limit]

        self.data_type = annFile.split("instances_")[-1].split(".json")[0]
        super(CocoDetection, self).__init__()

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        # image = cv2.imread(os.path.join(self.root, self.data_type, path))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return Image.open(os.path.join(self.root, self.data_type, path)).convert('RGB')

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        target = {'image_id': id, 'annotations': target}

        image, target = self.transform(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


@Registers.datasets.register("Coco")
class CocoDataSetDict(BaseDataSetDict):
    @classmethod
    def load(cls, train_limit=None, config=None):

        return cls({"train": CocoDetection(config.root_path, config.train_ann_path, limit=train_limit),
                    "test": CocoDetection(config.root_path, config.val_ann_path)})

    def get_object_detection_schema_dataset(self, dataset_type, config=None):

        if dataset_type == "train":
            dataset = self["train"]
        else:
            dataset = self["test"]

        return dataset


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_type):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_type = iou_type
        self.img_ids = []

        self.coco_eval = COCOeval(coco_gt, iouType=iou_type)
        self.eval_imgs = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        results = self.prepare(predictions, self.iou_type)

        # suppress pycocotools prints
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

        self.coco_eval.cocoDt = coco_dt
        self.coco_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = evaluate(self.coco_eval)

        self.eval_imgs.append(eval_imgs)

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def synchronize_between_processes(self):
        self.eval_imgs = np.concatenate(self.eval_imgs, 2)
        create_common_coco_eval(self.coco_eval, self.img_ids, self.eval_imgs)

    def accumulate(self):
        self.coco_eval.accumulate()

    def summarize(self):
        print("IoU metric: {}".format(self.iou_type))
        self.coco_eval.summarize()
        stats = self.coco_eval.stats
        return stats

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = tlx.convert_to_numpy(prediction["scores"]).tolist()
            labels = tlx.convert_to_numpy(prediction["labels"]).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask, dtype=np.uint8, order="F"))
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin = boxes[:, 0]
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]

    return tlx.convert_to_numpy(tlx.stack((xmin, ymin, xmax - xmin, ymax - ymin), axis=1))


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def all_gather(data):
    return [data]