import sys
import time
import torch
import torchvision
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from vehicle.ml.models.coco_eval import CocoEvaluator
from vehicle.exception import VehicleException
from vehicle.ml.models.utils import MetricLogger


def convert_to_coco_api(ds):
    try:
        coco_ds = COCO()
        # annotation IDs need to start at 1, not 0, see torchvision issue #1530
        ann_id = 1
        dataset = {"images": [], "categories": [], "annotations": []}
        categories = set()
        for img_idx in range(len(ds)):
            # find better way to get target
            # targets = ds.get_annotations(img_idx)
            img, targets = ds[img_idx]
            print(targets)
            image_id = targets["image_id"].item()
            img_dict={}
            img_dict["id"] = image_id
            img_dict["height"] = img.shape[-2]
            img_dict["width"] = img.shape[-1]
            dataset["images"].append(img_dict)
            bboxes = targets["boxes"].clone()
            bboxes[:, 2:] -= bboxes[:, :2]
            bboxes = bboxes.tolist()
            labels = targets["labels"].tolist()
            areas = targets["area"].tolist()
            iscrowd = targets["iscrowd"].tolist()
            if "masks" in targets:
                masks = targets["masks"]
                # make masks Fortran contiguous for coco_mask
                masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
            if "keypoints" in targets:
                keypoints = targets["keypoints"]
                keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
            num_objs = len(bboxes)
            for i in range(num_objs):
                ann = {}
                ann["image_id"] = image_id
                ann["bbox"] = bboxes[i]
                ann["category_id"] = labels[i]
                categories.add(labels[i])
                ann["area"] = areas[i]
                ann["iscrowd"] = iscrowd[i]
                ann["id"] = ann_id
                if "masks" in targets:
                    ann["segmentation"] = coco_mask.encode(masks[i].numpy())
                if "keypoints" in targets:
                    ann["keypoints"] = keypoints[i]
                    ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
                dataset["annotations"].append(ann)
                ann_id += 1
        dataset["categories"] = [{"id": i} for i in sorted(categories)]
        coco_ds.dataset = dataset
        coco_ds.createIndex()
        return coco_ds
    except Exception as e:
        raise VehicleException(e, sys) from e

def get_coco_api_from_dataset(dataset):
    try:
        for _ in range(10):
            if isinstance(dataset, torchvision.datasets.CocoDetection):
                break
            if isinstance(dataset, torch.utils.data.Subset):
                dataset = dataset.dataset
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            return dataset.coco
        return convert_to_coco_api(dataset)
    except Exception as e:
        raise VehicleException(e, sys) from e

def _get_iou_types(model):
    try:
        model_without_ddp = model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = model.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types
    except Exception as e:
        raise VehicleException(e, sys) from e

def evaluate(model, data_loader, device):
    try:
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"

        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        return coco_evaluator
    except Exception as e:
        raise VehicleException(e, sys) from e
