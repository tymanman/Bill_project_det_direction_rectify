# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from rectify_roi import *
from cls_inference import *
# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--classifier-weights")
    parser.add_argument("--root")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser



if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    root = args.root
    res_dir = args.output
    det_model = build_model(cfg)
    det_model.eval()
    checkpointer = DetectionCheckpointer(det_model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    cls_model = init_classifier(args.classifier_weights)
    aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
    for path in tqdm.tqdm(args.input):
        ori_img = read_image(os.path.join(root, path), format="BGR")
        start_time = time.time()
        height, width = ori_img.shape[:2]
        img = aug.get_transform(ori_img).apply_image(ori_img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        inputs = {"image": img, "height": height, "width": width}
        predictions = det_model([inputs])[0]["instances"]
        points = predictions.pred_points.cpu().detach().numpy()
        points = points.reshape(-1, 4, 2).astype( np.int32 )
        ori_img = ori_img.astype(np.uint8)
        rois_ = [project_rectify(ori_img, point) for point in points]
        shaped_rois = [cv2.resize(roi[:, :, ::-1], (500, 500)) for roi in rois_]
        rotations = cls_predict(cls_model, shaped_rois)
        rotated_rois = list(map(lambda index,roi: cv2.rotate(roi, cv2.ROTATE_180) if rotations[index] else roi, range(len(rois_)), rois_))
        if args.output:
            [cv2.imwrite(os.path.join(args.output, path.rsplit(".", 1)[0]+ \
            f"_{index}." + path.rsplit(".", 1)[-1]), roi) for index, roi in enumerate(rotated_rois)]
    #     if args.output:
    #         if os.path.isdir(args.output):
    #             assert os.path.isdir(args.output), args.output
    #             out_filename = os.path.join(args.output, os.path.basename(path))
    #         else:
    #             assert len(args.input) == 1, "Please specify a directory with args.output"
    #             out_filename = args.output
    #         visualized_output.save(out_filename)
    #     else:
    #         cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #         cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    #         if cv2.waitKey(0) == 27:
    #             break  # esc to quit
    # cv2.destroyAllWindows()
