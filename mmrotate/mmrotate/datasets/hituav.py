# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import warnings
import zipfile
from collections import defaultdict
from functools import partial
import json
import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset
from collections import OrderedDict

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS

@ROTATED_DATASETS.register_module()
class HITUAVDataset(CustomDataset):
    """HIT-UAV custom dataset with angle=0, support for data_prefix."""

    METAINFO = {
        'classes': ('person', 'car', 'bicycle', 'other vehicle'),
        'palette': [(0, 0, 255), (255, 0, 255), (255, 0, 0), (0, 255, 0)]
    }

    def __init__(self, ann_file, img_prefix, pipeline, **kwargs):
        # data_prefix: dict(img_path=..., etc)
        super(HITUAVDataset,self).__init__(ann_file=ann_file, img_prefix=img_prefix,pipeline=pipeline, **kwargs)

    def load_annotations(self, ann_folder):
        """Load annotation files from a folder (not a single json)."""
        data_infos = []
        class_map = {
            'person': 0,
            'car': 1,
            'bicycle': 2,
            'other vehicle': 3
        }

        json_files = sorted(f for f in os.listdir(ann_folder) if f.endswith('.json'))

        for idx, file_name in enumerate(json_files):
            json_path = os.path.join(ann_folder, file_name)
            with open(json_path, 'r') as f:
                ann_data = json.load(f)

            img_filename = file_name.replace('.json', '')  # '0_60_30_0_01611.jpg'

            width = ann_data['size']['width']
            height = ann_data['size']['height']

            gt_bboxes = []
            gt_labels = []

            for obj in ann_data['objects']:
                cls = obj['classTitle']
                if cls == 'dontcare' or cls not in class_map:
                    continue
                if obj['geometryType'] != 'rectangle':
                    continue

                x1, y1 = obj['points']['exterior'][0]
                x2, y2 = obj['points']['exterior'][1]
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                cx = x + w / 2
                cy = y + h / 2
                angle = 0.0  # axis-aligned

                gt_bboxes.append([cx, cy, w, h, angle])
                gt_labels.append(class_map[cls])

            data_infos.append(dict(
                filename=img_filename,
                width=width,
                height=height,
                ann=dict(
                    bboxes=np.array(gt_bboxes, dtype=np.float32) if gt_bboxes else np.zeros((0, 5), dtype=np.float32),
                    labels=np.array(gt_labels, dtype=np.int64) if gt_labels else np.array([], dtype=np.int64)
                )
            ))

        return data_infos

    def _filter_imgs(self):
        """
        Filter out images without valid annotations
        """
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if data_info['ann']['bboxes'].size > 0:
                valid_inds.append(i)
        return valid_inds

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 use_07_metric=True,
                 nproc=4
    ):
        """
        Evaluate dataset performance with mAP.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr

        if metric == 'mAP':
            assert isinstance(iou_thrs, list)

            mean_aps = []
            for iou_thr in iou_thrs:
                mmcv.print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')

                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    iou_thr=iou_thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc,
                    scale_ranges=scale_ranges)

                if isinstance(mean_ap, list):
                    for idx, ap in enumerate(mean_ap):  
                        eval_results[f'AP{int(iou_thr * 100):02d}_scale{idx}'] = ap
                else:
                    eval_results[f'AP{int(iou_thr * 100):02d}'] = mean_ap

                mean_aps.append(mean_ap)
                print(mean_aps)

            mean_aps = list(chain(*mean_aps)) if any(isinstance(i, list) for i in mean_aps) else mean_aps

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps) if mean_aps else 0.0
            eval_results.move_to_end('mAP', last=False)

        elif metric == 'recall':
            raise NotImplementedError

        return eval_results


