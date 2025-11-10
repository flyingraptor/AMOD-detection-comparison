# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile
import os
from ..builder import ROTATED_PIPELINES
from mmdet.core import BitmapMasks
import cv2

@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results

@ROTATED_PIPELINES.register_module()
class LoadSegmentMasks:
    """Load segmentation masks for AMOD dataset from .npy files.

    Accepts:
      - (C,H,W) uint8/float: channel-wise mask (>0 -> 1)
      - (H,W) semantic labels: each unique label (>0) -> one mask

    Outputs:
      results['gt_masks']: BitmapMasks((N,H,W) bool), resized to current image size
    """

    def __init__(self, mask_prefix, file_suffix='.npy', binarize_threshold=0):
        self.mask_prefix = mask_prefix
        self.file_suffix = file_suffix
        self.binarize_threshold = binarize_threshold

    def _standardize(self, arr: np.ndarray):
        """Return (N,H,W) uint8 {0,1} before resize."""
        arr = np.asarray(arr)
        if arr.ndim == 3:
            # assume (C,H,W) if first dim small
            if arr.shape[0] <= 64 and arr.shape[1] != arr.shape[-1]:
                masks = (arr > self.binarize_threshold).astype(np.uint8)  # (C,H,W)
            else:  # (H,W,C) → (C,H,W)
                masks = (np.transpose(arr, (2, 0, 1)) > self.binarize_threshold).astype(np.uint8)
            return masks
        elif arr.ndim == 2:
            labels = np.unique(arr)
            labels = labels[labels > self.binarize_threshold]
            if len(labels) == 0:
                H, W = arr.shape
                return np.zeros((0, H, W), dtype=np.uint8)
            masks = np.stack([(arr == lb).astype(np.uint8) for lb in labels], axis=0)  # (N,H,W)
            return masks
        else:
            raise ValueError(f"Unsupported mask shape: {arr.shape}")

    def __call__(self, results):
        # 현재 이미지 크기 (LoadImageFromFile 이후라면 존재)
        img = results.get('img', None)
        if img is not None:
            H, W = img.shape[:2]
        else:
            H, W = results['img_info']['height'], results['img_info']['width']

        img_filename = os.path.splitext(os.path.basename(results['img_info']['filename']))[0]
        mask_filename = f"Mask-{img_filename}{self.file_suffix}"
        mask_path = os.path.join(self.mask_prefix, mask_filename)

        if not os.path.exists(mask_path):
            empty = np.zeros((0, H, W), dtype=bool)
            results['gt_masks'] = BitmapMasks(empty, height=H, width=W)
            return results

        mask_array = np.load(mask_path)  # dtype 그대로
        inst = self._standardize(mask_array)  # (N,h0,w0)

        # 현재 이미지 크기로 리사이즈
        if inst.size == 0:
            resized = np.zeros((0, H, W), dtype=bool)
        else:
            resized = np.stack(
                [cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                 for m in inst],
                axis=0
            )

        results['gt_masks'] = BitmapMasks(resized, height=H, width=W)
        return results
