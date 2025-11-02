# Copyright (c) OpenMMLab. All rights reserved.
import copy

import cv2
import mmcv
import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmdet.datasets.pipelines.transforms import (Mosaic, RandomCrop,
                                                 RandomFlip, Resize)
from numpy import random

from mmrotate.core import norm_angle, obb2poly_np, poly2obb_np
from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class RResize(Resize):
    """Resize images & rotated bbox Inherit Resize pipeline class to handle
    rotated bboxes.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio).
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None):
        super(RResize, self).__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=True)

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            orig_shape = bboxes.shape
            bboxes = bboxes.reshape((-1, 5))
            w_scale, h_scale, _, _ = results['scale_factor']
            bboxes[:, 0] *= w_scale
            bboxes[:, 1] *= h_scale
            bboxes[:, 2:4] *= np.sqrt(w_scale * h_scale)
            results[key] = bboxes.reshape(orig_shape)


@ROTATED_PIPELINES.register_module()
class RRandomFlip(RandomFlip):
    """

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'.
        version (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self, flip_ratio=None, direction='horizontal', version='oc'):
        self.version = version
        super(RRandomFlip, self).__init__(flip_ratio, direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally or vertically.

        Args:
            bboxes(ndarray): shape (..., 5*k)
            img_shape(tuple): (height, width)

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        assert bboxes.shape[-1] % 5 == 0
        orig_shape = bboxes.shape
        bboxes = bboxes.reshape((-1, 5))
        flipped = bboxes.copy()
        if direction == 'horizontal':
            flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
        elif direction == 'vertical':
            flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
        elif direction == 'diagonal':
            flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
            flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
            return flipped.reshape(orig_shape)
        else:
            raise ValueError(f'Invalid flipping direction "{direction}"')
        if self.version == 'oc':
            rotated_flag = (bboxes[:, 4] != np.pi / 2)
            flipped[rotated_flag, 4] = np.pi / 2 - bboxes[rotated_flag, 4]
            flipped[rotated_flag, 2] = bboxes[rotated_flag, 3]
            flipped[rotated_flag, 3] = bboxes[rotated_flag, 2]
        else:
            flipped[:, 4] = norm_angle(np.pi - bboxes[:, 4], self.version)
        return flipped.reshape(orig_shape)


@ROTATED_PIPELINES.register_module()
class PolyRandomRotate(object):
    """Rotate img & bbox.
    Reference: https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA

    Args:
        rotate_ratio (float, optional): The rotating probability.
            Default: 0.5.
        mode (str, optional) : Indicates whether the angle is chosen in a
            random range (mode='range') or in a preset list of angles
            (mode='value'). Defaults to 'range'.
        angles_range(int|list[int], optional): The range of angles.
            If mode='range', angle_ranges is an int and the angle is chosen
            in (-angles_range, +angles_ranges).
            If mode='value', angles_range is a non-empty list of int and the
            angle is chosen in angles_range.
            Defaults to 180 as default mode is 'range'.
        auto_bound(bool, optional): whether to find the new width and height
            bounds.
        rect_classes (None|list, optional): Specifies classes that needs to
            be rotated by a multiple of 90 degrees.
        allow_negative (bool, optional): Whether to allow an image that does
            not contain any bbox area. Default False.
        version  (str, optional): Angle representations. Defaults to 'le90'.
    """

    def __init__(self,
                 rotate_ratio=0.5,
                 mode='range',
                 angles_range=180,
                 auto_bound=False,
                 rect_classes=None,
                 allow_negative=False,
                 version='le90'):
        self.rotate_ratio = rotate_ratio
        self.auto_bound = auto_bound
        assert mode in ['range', 'value'], \
            f"mode is supposed to be 'range' or 'value', but got {mode}."
        if mode == 'range':
            assert isinstance(angles_range, int), \
                "mode 'range' expects angle_range to be an int."
        else:
            assert mmcv.is_seq_of(angles_range, int) and len(angles_range), \
                "mode 'value' expects angle_range as a non-empty list of int."
        self.mode = mode
        self.angles_range = angles_range
        self.discrete_range = [90, 180, -90, -180]
        self.rect_classes = rect_classes
        self.allow_negative = allow_negative
        self.version = version

    @property
    def is_rotate(self):
        """Randomly decide whether to rotate."""
        return np.random.rand() < self.rotate_ratio

    def apply_image(self, img, bound_h, bound_w, interp=cv2.INTER_LINEAR):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0:
            return img
        return cv2.warpAffine(
            img, self.rm_image, (bound_w, bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y)
        points
        """
        if len(coords) == 0:
            return coords
        coords = np.asarray(coords, dtype=float)
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def create_rotation_matrix(self,
                               center,
                               angle,
                               bound_h,
                               bound_w,
                               offset=0):
        """Create rotation matrix."""
        center += offset
        rm = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        if self.auto_bound:
            rot_im_center = cv2.transform(center[None, None, :] + offset,
                                          rm)[0, 0, :]
            new_center = np.array([bound_w / 2, bound_h / 2
                                   ]) + offset - rot_im_center
            rm[:, 2] += new_center
        return rm

    def filter_border(self, bboxes, h, w):
        """Filter the box whose center point is outside or whose side length is
        less than 5."""
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        w_bbox, h_bbox = bboxes[:, 2], bboxes[:, 3]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) & \
                    (w_bbox > 5) & (h_bbox > 5)
        return keep_inds

    def __call__(self, results):
        """Call function of PolyRandomRotate."""
        if not self.is_rotate:
            results['rotate'] = False
            angle = 0
        else:
            results['rotate'] = True
            if self.mode == 'range':
                angle = self.angles_range * (2 * np.random.rand() - 1)
            else:
                i = np.random.randint(len(self.angles_range))
                angle = self.angles_range[i]

            class_labels = results['gt_labels']
            for classid in class_labels:
                if self.rect_classes:
                    if classid in self.rect_classes:
                        np.random.shuffle(self.discrete_range)
                        angle = self.discrete_range[0]
                        break

        h, w, c = results['img_shape']
        img = results['img']
        results['rotate_angle'] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = \
            abs(np.cos(angle / 180 * np.pi)), abs(np.sin(angle / 180 * np.pi))
        if self.auto_bound:
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos,
                 h * abs_cos + w * abs_sin]).astype(int)
        else:
            bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(image_center, angle,
                                                     bound_h, bound_w)
        self.rm_image = self.create_rotation_matrix(
            image_center, angle, bound_h, bound_w, offset=-0.5)

        img = self.apply_image(img, bound_h, bound_w)
        results['img'] = img
        results['img_shape'] = (bound_h, bound_w, c)
        gt_bboxes = results.get('gt_bboxes', [])
        labels = results.get('gt_labels', [])

        if len(gt_bboxes):
            gt_bboxes = np.concatenate(
                [gt_bboxes, np.zeros((gt_bboxes.shape[0], 1))], axis=-1)
            polys = obb2poly_np(gt_bboxes, self.version)[:, :-1].reshape(-1, 2)
            polys = self.apply_coords(polys).reshape(-1, 8)
            gt_bboxes = []
            for pt in polys:
                pt = np.array(pt, dtype=np.float32)
                obb = poly2obb_np(pt, self.version) \
                    if poly2obb_np(pt, self.version) is not None\
                    else [0, 0, 0, 0, 0]
                gt_bboxes.append(obb)
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            keep_inds = self.filter_border(gt_bboxes, bound_h, bound_w)
            gt_bboxes = gt_bboxes[keep_inds, :]
            labels = labels[keep_inds]
        if len(gt_bboxes) == 0 and not self.allow_negative:
            return None
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = labels

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_ratio={self.rotate_ratio}, ' \
                    f'base_angles={self.base_angles}, ' \
                    f'angles_range={self.angles_range}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class RRandomCrop(RandomCrop):
    """Random crop the image & bboxes.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        iof_thr (float): The minimal iof between a object and window.
            Defaults to 0.7.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels must be aligned. That is, `gt_bboxes`
          corresponds to `gt_labels`, and `gt_bboxes_ignore` corresponds to
          `gt_labels_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 iof_thr=0.7,
                 version='oc'):
        self.version = version
        self.iof_thr = iof_thr
        super(RRandomCrop, self).__init__(crop_size, crop_type,
                                          allow_negative_crop)

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('bbox_fields', []):
            assert results[key].shape[-1] % 5 == 0

        for key in results.get('img_fields', ['img']):
            img = results[key] #img
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        # results['img_shape'] = img_shape # e.g. (800, 800, 3)
        results['img_shape'] = img_shape[:2] # e.g. (800, 800)

        height, width, _ = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, 0, 0, 0],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            windows = np.array([width / 2, height / 2, width, height, 0],
                               dtype=np.float32).reshape(-1, 5)

            valid_inds = box_iou_rotated(
                torch.tensor(bboxes), torch.tensor(windows),
                mode='iof').numpy().squeeze() > self.iof_thr

            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None

            valid_inds = np.atleast_1d(valid_inds)

            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]
        return results


@ROTATED_PIPELINES.register_module()
class RMosaic(Mosaic):
    """Rotate Mosaic augmentation. Inherit from
    `mmdet.datasets.pipelines.transforms.Mosaic`.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text
                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:
         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Defaults to 0.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` is invalid. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        version  (str, optional): Angle representations. Defaults to `oc`.
    """

    def __init__(self,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=10,
                 bbox_clip_border=True,
                 skip_filter=True,
                 pad_val=114,
                 prob=1.0,
                 version='oc'):
        super(RMosaic, self).__init__(
            img_scale=img_scale,
            center_ratio_range=center_ratio_range,
            min_bbox_size=min_bbox_size,
            bbox_clip_border=bbox_clip_border,
            skip_filter=skip_filter,
            pad_val=pad_val,
            prob=1.0)

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0] = \
                    scale_ratio_i * gt_bboxes_i[:, 0] + padw
                gt_bboxes_i[:, 1] = \
                    scale_ratio_i * gt_bboxes_i[:, 1] + padh
                gt_bboxes_i[:, 2:4] = \
                    scale_ratio_i * gt_bboxes_i[:, 2:4]

            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            mosaic_bboxes, mosaic_labels = \
                self._filter_box_candidates(
                    mosaic_bboxes, mosaic_labels,
                    2 * self.img_scale[1], 2 * self.img_scale[0]
                )
        # If results after rmosaic does not contain any valid gt-bbox,
        # return None. And transform flows in MultiImageMixDataset will
        # repeat until existing valid gt-bbox.
        if len(mosaic_bboxes) == 0:
            return None

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels

        return results

    def _filter_box_candidates(self, bboxes, labels, w, h):
        """Filter out small bboxes and outside bboxes after Mosaic."""
        bbox_x, bbox_y, bbox_w, bbox_h = \
            bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        valid_inds = (bbox_x > 0) & (bbox_x < w) & \
                     (bbox_y > 0) & (bbox_y < h) & \
                     (bbox_w > self.min_bbox_size) & \
                     (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds]

@ROTATED_PIPELINES.register_module()
class MaskBrightness:
    """
    Object-aware brightness augmentation (두 모드만 지원)
    - mode: 'perlin' | 'simplex'
      * perlin  : 저해상도 랜덤 필드 upsample + GaussianBlur (low-frequency field)
      * simplex : 2D Simplex noise (Ken Perlin, 2001) 기반 low-frequency field
    """
    def __init__(self,
                 strength=0.25,
                 blur_sigma=40,
                 feather_px=5,
                 per_instance=True,
                 preserve_mean=True,
                 prob=0.7,
                 seed=None,
                 mode='perlin'):
        self.strength = float(strength)
        self.blur_sigma = int(blur_sigma)
        self.feather_px = int(feather_px)
        self.per_instance = bool(per_instance)
        self.preserve_mean = bool(preserve_mean)
        self.prob = float(prob)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mode = mode

        # Simplex용 permutation 테이블(재현성; reproducibility)
        self._perm256 = np.arange(256, dtype=np.int32)
        self.rng.shuffle(self._perm256)
        self._perm512 = np.concatenate([self._perm256, self._perm256])

    # ------------------------------
    # 2D Simplex noise (returns [-1, 1])
    # ------------------------------
    def _simplex2d(self, x, y):
        F2 = 0.5 * (np.sqrt(3.0) - 1.0)
        G2 = (3.0 - np.sqrt(3.0)) / 6.0

        s = (x + y) * F2
        i = np.floor(x + s).astype(np.int32)
        j = np.floor(y + s).astype(np.int32)

        t = (i + j) * G2
        X0 = i - t
        Y0 = j - t
        x0 = x - X0
        y0 = y - Y0

        i1 = (x0 > y0).astype(np.int32)
        j1 = 1 - i1

        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2

        ii = (i & 255)
        jj = (j & 255)
        perm = self._perm512

        gi0 = perm[ii + perm[jj]] % 12
        gi1 = perm[(ii + i1) + perm[(jj + j1)]] % 12
        gi2 = perm[(ii + 1) + perm[(jj + 1)]] % 12

        grads = np.array([
            [ 1,  1], [-1,  1], [ 1, -1], [-1, -1],
            [ 1,  0], [-1,  0], [ 1,  0], [-1,  0],
            [ 0,  1], [ 0, -1], [ 0,  1], [ 0, -1]
        ], dtype=np.float32)

        def contrib(t, gx, gy, dx, dy):
            mask = t > 0
            t2 = np.where(mask, t * t, 0.0)
            t4 = t2 * t2
            dot = gx * dx + gy * dy
            return np.where(mask, t4 * dot, 0.0)

        t0 = 0.5 - x0 * x0 - y0 * y0
        t1 = 0.5 - x1 * x1 - y1 * y1
        t2 = 0.5 - x2 * x2 - y2 * y2

        g0 = grads[gi0]
        g1 = grads[gi1]
        g2 = grads[gi2]

        n0 = contrib(t0, g0[..., 0], g0[..., 1], x0, y0)
        n1 = contrib(t1, g1[..., 0], g1[..., 1], x1, y1)
        n2 = contrib(t2, g2[..., 0], g2[..., 1], x2, y2)

        noise = 70.0 * (n0 + n1 + n2)  # normalization factor
        noise = np.clip(noise, -1.5, 1.5)

        mmin, mmax = float(noise.min()), float(noise.max())
        if mmax - mmin < 1e-6:
            return np.zeros_like(noise, dtype=np.float32)
        return (2.0 * (noise - mmin) / (mmax - mmin) - 1.0).astype(np.float32)

    def _simplex_field(self, h, w):
        """저주파(Simplex) 필드 생성 in [-1,1]. blur_sigma를 scale로 해석."""
        base = max(8.0, float(self.blur_sigma))
        freq = 1.0 / base
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32),
                             indexing='ij')
        return self._simplex2d(xx * freq, yy * freq)

    def _perlinlike_field(self, h, w):
        """저해상도 → upsample → GaussianBlur ([-1,1])"""
        sh = max(1, h // 8 + 1)
        sw = max(1, w // 8 + 1)
        small = self.rng.normal(0, 1, size=(sh, sw)).astype(np.float32)
        field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        if self.blur_sigma > 0:
            field = cv2.GaussianBlur(field, (0, 0), self.blur_sigma)

        fmin, fmax = float(field.min()), float(field.max())
        if fmax - fmin < 1e-6:
            return np.zeros((h, w), np.float32)
        return (2.0 * (field - fmin) / (fmax - fmin) - 1.0).astype(np.float32)

    def _lowfreq_field(self, h, w):
        if self.mode == 'perlin':
            return self._perlinlike_field(h, w)
        elif self.mode == 'simplex':
            return self._simplex_field(h, w)
        else:
            raise ValueError("mode must be one of {'perlin', 'simplex'}")

    def __call__(self, results):
        if self.rng.random() > self.prob:
            return results

        gt_masks = results.get('gt_masks', None)
        if gt_masks is None:
            return results

        img = results['img']
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        out = img.astype(np.float32).copy()
        masks = gt_masks.to_ndarray().astype(np.uint8)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if not self.per_instance:
            masks = masks.max(axis=0, keepdims=True)

        H, W = img.shape[:2]

        for mk in masks:
            if mk.max() == 0:
                continue

            field = self._lowfreq_field(H, W)

            if self.preserve_mean and mk.sum() > 0:
                field = field - field[mk > 0].mean()

            scale = 1.0 + self.strength * field
            trans = out * scale[..., None]

            if self.feather_px > 0:
                soft = cv2.GaussianBlur(
                    mk * 255,
                    (self.feather_px * 2 + 1, self.feather_px * 2 + 1),
                    0
                ).astype(np.float32) / 255.0
            else:
                soft = mk.astype(np.float32)

            soft3 = soft[..., None]
            out = trans * soft3 + out * (1.0 - soft3)

        results['img'] = np.clip(out, 0, 255).astype(np.uint8)
        return results


@ROTATED_PIPELINES.register_module()
class MaskContourBlur:
    """Object-aware contour blur augmentation.

    Args:
        blur_ksize (int): 블러 커널 크기 (홀수 권장, default=15)
        contour_thickness (int): 외곽선 두께 (px)
        per_instance (bool): 객체별로 독립 처리 여부
        prob (float): 적용 확률
        seed (int | None): 랜덤 시드
    """

    def __init__(self,
                 blur_ksize=10,
                 contour_thickness=3,
                 per_instance=True,
                 prob=0.5,
                 seed=None):
        self.blur_ksize = int(blur_ksize) | 1  # 홀수 보장
        self.contour_thickness = int(contour_thickness)
        self.per_instance = bool(per_instance)
        self.prob = float(prob)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, results):
        if self.rng.random() > self.prob:
            return results

        gt_masks = results.get('gt_masks', None)
        if gt_masks is None:
            return results

        img = results['img']
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        out = img.astype(np.float32).copy()
        masks = gt_masks.to_ndarray().astype(np.uint8)

        if masks.ndim == 2:  # 단일 객체
            masks = masks[None, ...]

        if not self.per_instance:
            masks = masks.max(axis=0, keepdims=True)

        blurred = cv2.GaussianBlur(img, (self.blur_ksize, self.blur_ksize), 0)

        for mk in masks:
            if mk.max() == 0:
                continue

            mask_bin = (mk > 0).astype(np.uint8) * 255

            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contour_mask = np.zeros_like(mask_bin)
            cv2.drawContours(contour_mask, contours, -1, 255, thickness=self.contour_thickness)

            contour_mask_3 = np.repeat(contour_mask[..., None], 3, axis=2) / 255.0

            out = blurred * contour_mask_3 + out * (1.0 - contour_mask_3)

        results['img'] = np.clip(out, 0, 255).astype(np.uint8)
        return results


@ROTATED_PIPELINES.register_module()
class MaskBrightness:
    """
    Object-aware brightness augmentation (unified modes)

    mode in {
      'gaussian', 'average', 'median', 'fft', 'perlin', 'simplex'
    }

    모든 모드는 동일한 규약으로 저주파 필드를 생성:
      field(H, W; sigma_like) -> float32 in [-1, 1]
    """

    def __init__(self,
                 strength=0.25,
                 blur_sigma=40,       
                 feather_px=5,
                 per_instance=True,
                 preserve_mean=True,
                 prob=0.7,
                 seed=None,
                 mode='gaussian'):
        
        self.strength = float(strength) # 객체 내부 밝기 변조의 강도 0.25 면, 최대 1.25 최소 -1.25 까지 변화 가능
        self.blur_sigma = float(blur_sigma)  # 노이즈 필드의 공간 스케일 역할 , 값이 클수록 더 완만하고 큰 스케일 변화 
        self.feather_px = int(feather_px) # 변조와 함께 가장자리를 부드럽게 하여 , 자연스럽게 연결
        self.per_instance = bool(per_instance) # True 면 객체별로 독립된 필드 생성 , False 면 이미지 전체에 대해 하나의 필드 사용
        self.preserve_mean = bool(preserve_mean) # True 면 밝기 변조 후에도 평균 밝기 유지 , False 면 단순 변조
        self.prob = float(prob) # 증강을 적용할 확률 
        self.seed = seed # 난수 생성기 
        self.rng = np.random.default_rng(seed)
        self.mode = str(mode).lower() # 필드 모드 선택


        # Simplex용 permutation (reproducibility)
        self._perm256 = np.arange(256, dtype=np.int32)
        self.rng.shuffle(self._perm256)
        self._perm512 = np.concatenate([self._perm256, self._perm256])

        # Simplex용 고정 gradients (12)
        self._grads12 = np.array([
            [ 1,  1], [-1,  1], [ 1, -1], [-1, -1],
            [ 1,  0], [-1,  0], [ 1,  0], [-1,  0],
            [ 0,  1], [ 0, -1], [ 0,  1], [ 0, -1]
        ], dtype=np.float32)


    # -------------------------------------------------------------------
    # blur_sigma 변수를 모든 블러/필드 모드에 사용할 수 있게 매핑하는 함수들
    # -------------------------------------------------------------------

    # 입력한 필드를 [-1,1] 범위로 정규화함
    @staticmethod
    def _normalize(field: np.ndarray) -> np.ndarray:
        field = field.astype(np.float32, copy=False)
        fmin, fmax = float(field.min()), float(field.max())
        if fmax - fmin < 1e-6:
            return np.zeros_like(field, dtype=np.float32)
        return (2.0 * (field - fmin) / (fmax - fmin) - 1.0).astype(np.float32)

    # average/median 용 openCV 처리를 위한 홀수 커널 크기 변환
    @staticmethod
    def _sigma_to_kernel(s: float) -> int:
        s = max(1.0, float(s))
        k = int(np.floor(6.0 * s + 1))
        if k % 2 == 0:
            k += 1
        k = max(3, min(k, 255))  # OpenCV 제약
        return k

    # FFT 기반 low-pass 필터에서 사용할 원형 마스크 반경을 시그마로 부터 계산
    @staticmethod
    def _sigma_to_radius(s: float) -> int:
        """sigma_like -> FFT 원형 low-pass 반경."""
        s = max(1.0, float(s))
        # 간단 스케일 매핑: radius ≈ 3*s
        r = int(np.clip(np.round(3.0 * s), 1, 10_000))
        return r

    # ------------------------------
    # 각 모드의 통일된 생성자 (Unified generators)
    # ------------------------------
    def _gen_gaussian(self, h, w, s):
        base = self.rng.normal(0, 1, size=(h, w)).astype(np.float32)
        field = cv2.GaussianBlur(base, (0, 0), s)
        return self._normalize(field)

    def _gen_average(self, h, w, s):
        base = self.rng.normal(0, 1, size=(h, w)).astype(np.float32)
        k = self._sigma_to_kernel(s)
        field = cv2.blur(base, (k, k))
        return self._normalize(field)

    def _gen_median(self, h, w, s):
        base = self.rng.normal(0, 1, size=(h, w)).astype(np.float32)
        k = self._sigma_to_kernel(s)
        # medianBlur는 float32 불가 → uint8 왕복, 이후 공통 normalize
        bmin, bmax = float(base.min()), float(base.max())
        if bmax - bmin < 1e-6:
            return np.zeros((h, w), dtype=np.float32)
        u8 = np.clip((base - bmin) / (bmax - bmin) * 255.0, 0, 255).astype(np.uint8)
        u8 = cv2.medianBlur(u8, k)
        field = u8.astype(np.float32)
        return self._normalize(field)

    def _gen_fft(self, h, w, s):
        base = self.rng.normal(0, 1, size=(h, w)).astype(np.float32)
        f = np.fft.fft2(base)
        fshift = np.fft.fftshift(f)
        crow, ccol = h // 2, w // 2
        r = self._sigma_to_radius(s)
        y, x = np.ogrid[:h, :w]
        mask = ((x - ccol) ** 2 + (y - crow) ** 2) <= r * r
        fshift = fshift * mask
        field = np.fft.ifft2(np.fft.ifftshift(fshift)).real.astype(np.float32)
        return self._normalize(field)

    def _gen_perlin(self, h, w, s):
        """Upsample noise + GaussianBlur; s는 공간 스케일."""
        # 저해상도 크기: s가 클수록 더 저주파 → 더 작은 샘플 크기
        down = int(max(4, min(64, np.round(max(h, w) / max(8.0, s*2.0)))))
        sh = max(2, h // down + 1)
        sw = max(2, w // down + 1)
        small = self.rng.normal(0, 1, size=(sh, sw)).astype(np.float32)
        field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        field = cv2.GaussianBlur(field, (0, 0), s)
        return self._normalize(field)

    # ---- Simplex 2D ----
    def _simplex2d(self, x, y):
        F2 = 0.5 * (np.sqrt(3.0) - 1.0)
        G2 = (3.0 - np.sqrt(3.0)) / 6.0

        s = (x + y) * F2
        i = np.floor(x + s).astype(np.int32)
        j = np.floor(y + s).astype(np.int32)

        t = (i + j) * G2
        X0 = i - t
        Y0 = j - t
        x0 = x - X0
        y0 = y - Y0

        i1 = (x0 > y0).astype(np.int32)
        j1 = 1 - i1

        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2

        ii = (i & 255)
        jj = (j & 255)
        perm = self._perm512

        gi0 = perm[ii + perm[jj]] % 12
        gi1 = perm[(ii + i1) + perm[(jj + j1)]] % 12
        gi2 = perm[(ii + 1) + perm[(jj + 1)]] % 12

        g0 = self._grads12[gi0]
        g1 = self._grads12[gi1]
        g2 = self._grads12[gi2]

        def contrib(t, gx, gy, dx, dy):
            mask = t > 0
            t2 = np.where(mask, t * t, 0.0)
            t4 = t2 * t2
            dot = gx * dx + gy * dy
            return np.where(mask, t4 * dot, 0.0)

        t0 = 0.5 - x0 * x0 - y0 * y0
        t1 = 0.5 - x1 * x1 - y1 * y1
        t2 = 0.5 - x2 * x2 - y2 * y2

        n0 = contrib(t0, g0[..., 0], g0[..., 1], x0, y0)
        n1 = contrib(t1, g1[..., 0], g1[..., 1], x1, y1)
        n2 = contrib(t2, g2[..., 0], g2[..., 1], x2, y2)

        noise = 70.0 * (n0 + n1 + n2)
        noise = np.clip(noise, -1.5, 1.5)
        return noise.astype(np.float32)

    def _gen_simplex(self, h, w, s):
        """Simplex noise; s는 공간 스케일 → freq = 1/s."""
        base = max(8.0, float(s))
        freq = 1.0 / base
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                             np.arange(w, dtype=np.float32),
                             indexing='ij')
        raw = self._simplex2d(xx * freq, yy * freq)
        return self._normalize(raw)

    # ------------------------------
    # 통일된 디스패처 (Unified dispatcher)
    # ------------------------------
    def _lowfreq_field(self, h, w):
        s = self.blur_sigma  # 공통 스케일 파라미터
        m = self.mode
        if m == 'gaussian':
            return self._gen_gaussian(h, w, s)
        elif m == 'average':
            return self._gen_average(h, w, s)
        elif m == 'median':
            return self._gen_median(h, w, s)
        elif m == 'fft':
            return self._gen_fft(h, w, s)
        elif m == 'perlin':
            return self._gen_perlin(h, w, s)
        elif m == 'simplex':
            return self._gen_simplex(h, w, s)
        else:
            raise ValueError("mode must be one of "
                             "{'gaussian','average','median','fft','perlin','simplex'}")

    # ------------------------------
    # Transform 본체
    # ------------------------------
    def __call__(self, results):
        if self.rng.random() > self.prob:
            return results

        gt_masks = results.get('gt_masks', None)
        if gt_masks is None:
            return results

        img = results['img']
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        out = img.astype(np.float32).copy()
        masks = gt_masks.to_ndarray().astype(np.uint8)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if not self.per_instance:
            masks = masks.max(axis=0, keepdims=True)

        H, W = img.shape[:2]

        for mk in masks:
            if mk.max() == 0:
                continue

            field = self._lowfreq_field(H, W)

            if self.preserve_mean and mk.sum() > 0:
                field = field - field[mk > 0].mean()

            scale = 1.0 + self.strength * field
            trans = out * scale[..., None]

            # Feathered soft mask
            if self.feather_px > 0:
                soft = cv2.GaussianBlur(
                    mk * 255,
                    (self.feather_px * 2 + 1, self.feather_px * 2 + 1),
                    0
                ).astype(np.float32) / 255.0
            else:
                soft = mk.astype(np.float32)

            soft3 = soft[..., None]
            out = trans * soft3 + out * (1.0 - soft3)

        results['img'] = np.clip(out, 0, 255).astype(np.uint8)
        return results


@ROTATED_PIPELINES.register_module()
class MaskContourBlur:
    """Object-aware contour blur augmentation.

    Args:
        blur_ksize (int): 블러 커널 크기 (홀수 권장)
        contour_thickness (tuple[int,int]): (min, max) 두께 범위(px)
        per_instance (bool): True면 객체별 독립 처리 (각 객체마다 두께 랜덤 샘플링)
        prob (float): 적용 확률
        seed (int | None): 랜덤 시드
    """

    def __init__(self,
                 blur_ksize=10,
                 contour_thickness=(2, 5),
                 per_instance=True,
                 prob=0.5,
                 seed=None):
        
        self.blur_ksize = int(blur_ksize) | 1  # 홀수 보장
        assert isinstance(contour_thickness, (tuple, list)) and len(contour_thickness) == 2, \
            "contour_thickness는 (min, max) 형태여야 합니다."
        self.tmin, self.tmax = map(int, contour_thickness)

        self.per_instance = bool(per_instance)
        self.prob = float(prob)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _sample_thickness(self) -> int:
        """두께 샘플링: min~max 범위 내 랜덤 정수"""
        return int(self.rng.integers(self.tmin, self.tmax + 1))

    def __call__(self, results):
        # 확률 체크
        if self.rng.random() > self.prob:
            return results

        gt_masks = results.get('gt_masks', None)
        if gt_masks is None:
            return results

        img = results['img']
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        out = img.astype(np.float32).copy()
        masks = gt_masks.to_ndarray().astype(np.uint8)

        if masks.ndim == 2:
            masks = masks[None, ...]

        # per_instance=False → 모든 객체를 합쳐 한 번만 처리
        if not self.per_instance:
            masks = masks.max(axis=0, keepdims=True)
            thickness = self._sample_thickness()  # 한 번만 샘플링
        else:
            thickness = None  # 각 객체마다 새로 샘플링

        # 전체 블러 사전 계산
        blurred = cv2.GaussianBlur(img, (self.blur_ksize, self.blur_ksize), 0)

        # 객체별 처리
        for mk in masks:
            if mk.max() == 0:
                continue

            mask_bin = (mk > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # per_instance=True → 객체마다 샘플링
            t = self._sample_thickness() if self.per_instance else thickness

            contour_mask = np.zeros_like(mask_bin)
            cv2.drawContours(contour_mask, contours, -1, 255, thickness=t)

            contour_mask_3 = np.repeat(contour_mask[..., None], 3, axis=2).astype(np.float32) / 255.0
            out = blurred * contour_mask_3 + out * (1.0 - contour_mask_3)

        results['img'] = np.clip(out, 0, 255).astype(np.uint8)
        return results
