# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile
import os
from ..builder import ROTATED_PIPELINES
from mmdet.core import BitmapMasks


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
    
    Each mask file is named 'Mask-{img_name}.npy'
    Example: image 'EO_0000_0.png' -> mask 'Mask-EO_0000_0.npy'
    """

    def __init__(self, mask_prefix, file_suffix='.npy'):
        self.mask_prefix = mask_prefix
        self.file_suffix = file_suffix

    def __call__(self, results):
        img_filename = os.path.splitext(os.path.basename(results['img_info']['filename']))[0]
        mask_filename = f"Mask-{img_filename}{self.file_suffix}"
        mask_path = os.path.join(self.mask_prefix, mask_filename)

        if not os.path.exists(mask_path):
            results['gt_masks'] = None
            return results

        mask_array = np.load(mask_path).astype(np.uint8)

        gt_masks = BitmapMasks(mask_array, height=mask_array.shape[1], width=mask_array.shape[2])
        results['gt_masks'] = gt_masks
        return results
