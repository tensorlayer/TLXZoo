import contextlib
import copy
import os

import numpy as np
import tensorlayerx as tlx
from PIL import Image
from pycocotools.coco import COCO
from tensorlayerx.dataflow import Dataset
from tensorlayerx.vision.transforms.utils import load_image


class CocoDetectionDataset(Dataset):
    def __init__(
        self, root, split='train', transform=None, image_format='pil'
    ):
        if split == 'train':
            ann_file = os.path.join(root, 'annotations/instances_train2017.json')
        else:
            ann_file = os.path.join(root, 'annotations/instances_val2017.json')
        self.coco = COCO(ann_file)
        self.root = root
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))
        # clear 0 label
        new_ids = []
        for id in self.ids:
            target = self._load_target(id)
            anno = [obj for obj in target if "iscrowd" not in obj or obj["iscrowd"] == 0]
            if len(anno) == 0:
                continue
            new_ids.append(id)
        self.ids = new_ids
        self.image_format = image_format

        print("load ids:", len(self.ids))

        self.data_type = ann_file.split("instances_")[-1].split(".json")[0]

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        if self.image_format == 'opencv':
            return load_image(os.path.join(self.root, self.data_type, path))
        else:
            return Image.open(os.path.join(self.root, self.data_type, path)).convert('RGB')

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        path = self.coco.loadImgs(id)[0]["file_name"]
        data = {'image_id': id, 'annotations': target, "path": os.path.join(self.root, self.data_type, path), 'image': image}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.ids)


class CocoHumanPoseEstimationDataset(Dataset):
    def __init__(
        self, root, split='train', transform=None, image_format='pil'
    ):
        if split == 'train':
            ann_file = os.path.join(root, 'annotations/person_keypoints_train2017.json')
        else:
            ann_file = os.path.join(root, 'annotations/person_keypoints_val2017.json')
        self.coco = COCO(ann_file)
        self.root = root
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))

        new_ids = []
        for id in self.ids:
            target = self._load_target(id)
            if not target:
                continue

            for index, t in enumerate(target):
                keypoints = t["keypoints"]
                if sum(keypoints) == 0:
                    continue
                new_ids.append((id, index))
        self.ids = new_ids

        self.image_format = image_format

        print("load ids:", len(self.ids))

        self.data_type = ann_file.split("person_keypoints_")[-1].split(".json")[0]

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        if self.image_format == 'opencv':
            return load_image(os.path.join(self.root, self.data_type, path))
        else:
            return Image.open(os.path.join(self.root, self.data_type, path)).convert('RGB')

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int):
        id, index = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)[index]
        text = os.path.join(self.root, self.data_type, self.coco.loadImgs(id)[0]["file_name"]) + " "
        text += str(image.height) + " "
        text += str(image.width) + " "
        text += " ".join([str(i) for i in target["bbox"]]) + " "
        text += " ".join([str(i) for i in target["keypoints"]]) + " "

        data = {'image_id': id, 'annotations': target, "text": text.strip(), 'image': image}

        if self.transform:
             data = self.transform(data)

        return data

    def __len__(self) -> int:
        return len(self.ids)


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_type):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_type = iou_type
        self.img_ids = []
        from pycocotools.cocoeval import COCOeval
        self.coco_eval = COCOeval(coco_gt, iouType=iou_type)
        self.eval_imgs = []

    def update(self, predictions):
        from pycocotools.coco import COCO
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
        import pycocotools.mask as mask_util
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