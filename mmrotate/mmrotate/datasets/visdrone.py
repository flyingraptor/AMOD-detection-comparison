import os
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv import print_log
from mmdet.datasets import CustomDataset
from PIL import Image

from mmrotate.core import eval_rbbox_map, poly2obb_np
from mmrotate.datasets.builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class VisDroneDataset(CustomDataset):
    """VisDrone dataset (rotated) for mmrotate.

    Expected directory example:
        root/
          └─ train/
              ├─ trainimg/        # img_prefix
              │    ├─ 00001.jpg
              │    └─ ...
              ├─ trainlabel/      # annot_prefix
              │    ├─ 00001.xml
              │    └─ ...
              └─ train.json        # ann_file: list of image ids (no extension)

    Args:
        ann_file (str): Path to a txt file which lists image ids (one per line).
        pipeline (list[dict]): Processing pipeline.
        version (str): Angle representation for poly2obb. Default: 'oc'.
        xmltype (str): 'obb' (polygon/robndbox) or 'hbb' (bndbox). Default: 'obb'.
        img_prefix (str): Directory containing images.
        annot_prefix (str): Directory containing xml annotations.
    """

    # VisDrone 10-class setup (common)
    CLASSES = (
        'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    )

    PALETTE = [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
        (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
        (0, 0, 192), (250, 170, 30)
    ]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 xmltype='obb',
                 img_prefix=None,
                 annot_prefix=None,
                 **kwargs):
        
        assert xmltype in ['hbb', 'obb']
        self.version = version
        self.xmltype = xmltype
        self.img_prefix = img_prefix
        self.annot_prefix = annot_prefix
        self.cat2label = {c: i for i, c in enumerate(self.CLASSES)}

        # 선택: 라벨 표기 변형/오타를 정규화하고 싶으면 여기에 매핑 추가
        self.alias_map = {
            'awning_tricycle': 'awning-tricycle',
            'motorcycle': 'motor',
            'pedestrain': 'pedestrian',
            'feright car': 'truck',            # 사용자 데이터 예외 처리 (원하면 수정)
            'freight car': 'truck',
            'freight_car': 'truck',
        }
        self._json_meta = {}  # NEW: {"00001": {"filename":"00001.jpg","width":640,"height":512}}
        super(VisDroneDataset, self).__init__(
            ann_file=ann_file, pipeline=pipeline, img_prefix=img_prefix, **kwargs
        )

    # --------- helpers ---------
    def _norm_class_name(self, cls: str) -> str:
        cls = cls.strip().lower()
        return self.alias_map.get(cls, cls)

    def _find_image_file(self, img_id: str):
        """Find actual image file by trying common extensions."""
        exts = ['.jpg', '.JPG', '.jpeg', '.png', '.PNG', '.bmp']
        for ext in exts:
            p = osp.join(self.img_prefix, img_id + ext)
            if osp.exists(p):
                return p, img_id + ext
        # fallback: if filename tag is provided in xml (handled later)
        return None, None

    def _parse_polygon(self, obj):
        """Parse polygon from XML:
           <polygon><x1>..</x1><y1>..</y1> ... (assume 4 points)"""
        poly = []
        # try x1..x4, y1..y4
        for k in range(1, 5):
            x_tag = obj.find(f'polygon/x{k}')
            y_tag = obj.find(f'polygon/y{k}')
            if x_tag is None or y_tag is None:
                return None
            poly.extend([float(x_tag.text), float(y_tag.text)])
        return np.array(poly, dtype=np.float32)  # [x1,y1,x2,y2,x3,y3,x4,y4]

    def _hbb_to_poly(self, bnd_box):
        """bndbox -> polygon in clockwise order."""
        xmin = float(bnd_box.find('xmin').text)
        ymin = float(bnd_box.find('ymin').text)
        xmax = float(bnd_box.find('xmax').text)
        ymax = float(bnd_box.find('ymax').text)
        return np.array([
            xmin, ymin,
            xmax, ymin,
            xmax, ymax,
            xmin, ymax
        ], dtype=np.float32)
    
    def _read_img_ids(self, ann_file):
        """TXT/JSON 인덱스 자동 파싱 → List[str] (확장자 제거된 ID) 반환"""
        _, ext = osp.splitext(ann_file)
        ext = ext.lower()
        if ext == '.json':
            data = mmcv.load(ann_file)
            img_ids = []

            # COCO-style: {"images":[{"file_name":..., "width":..., "height":...}, ...]}
            if isinstance(data, dict) and isinstance(data.get('images'), list):
                for im in data['images']:
                    fname = im.get('file_name') or im.get('filename')
                    if not fname:
                        continue
                    base = osp.splitext(osp.basename(fname))[0]
                    img_ids.append(base)
                    self._json_meta[base] = {
                        'filename': osp.basename(fname),
                        'width': int(im.get('width')) if im.get('width') is not None else None,
                        'height': int(im.get('height')) if im.get('height') is not None else None,
                    }
                img_ids = [i for i in img_ids if i]
                if not img_ids:
                    raise ValueError(f"[VisDroneDataset] No images parsed from JSON: {ann_file}")
                return img_ids

            # 보조 형식들 (옵션)
            if isinstance(data, list):
                return [osp.splitext(osp.basename(x))[0] for x in data if isinstance(x, str)]
            if isinstance(data, dict) and 'ids' in data:
                return [osp.splitext(osp.basename(str(x)))[0] for x in data['ids']]

            raise ValueError(f"[VisDroneDataset] Unsupported JSON format: {ann_file}")

        # fallback: .txt (기존 방식)
        return [x.strip() for x in mmcv.list_from_file(ann_file) if x.strip()]

    # --------- core API ---------
    def load_annotations(self, ann_file):
        """Load annotations from .txt or .json index."""
        data_infos = []
        img_ids = self._read_img_ids(ann_file)   # ← NEW: txt/json 자동 처리

        for img_id in img_ids:
            # JSON 메타 있으면 우선 사용 (filename, width/height)
            meta = self._json_meta.get(img_id)
            preferred_name = meta['filename'] if meta and meta.get('filename') else None

            # image path
            if preferred_name:
                img_path, filename = self._find_image_file(osp.splitext(preferred_name)[0])
            else:
                img_path, filename = self._find_image_file(img_id)

            # xml path (id 우선)
            xml_path = osp.join(self.annot_prefix, f'{img_id}.xml')
            if not osp.exists(xml_path):
                print_log(f'[VisDroneDataset] XML not found: {xml_path}', logger='current')
                # 파일명 기반 보조 매칭
                if filename is not None:
                    base = osp.splitext(filename)[0]
                    alt_xml = osp.join(self.annot_prefix, f'{base}.xml')
                    if osp.exists(alt_xml):
                        xml_path = alt_xml
                    else:
                        continue
                else:
                    continue

            # XML 파싱
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # filename 보완: XML <filename> → fallback
            if img_path is None:
                name_tag = root.find('filename')
                if name_tag is not None:
                    base = name_tag.text.strip()
                    img_path, filename = self._find_image_file(osp.splitext(base)[0])
                if img_path is None:
                    fallback = osp.join(self.img_prefix, img_id + '.jpg')
                    img_path, filename = fallback, osp.basename(fallback)

            # --- 여기부터 "크기 결정" 우선순위만 변경: XML → JSON(meta) → 이미지 오픈 ---
            width = height = None
            size_tag = root.find('size')
            if size_tag is not None:
                w = size_tag.find('width')
                h = size_tag.find('height')
                if w is not None and h is not None:
                    width = int(float(w.text))
                    height = int(float(h.text))

            if (width is None or height is None) and meta is not None:
                width = width or meta.get('width')
                height = height or meta.get('height')

            if width is None or height is None:
                with Image.open(img_path) as im:
                    width, height = im.size

            data_info = dict(
                filename=filename,
                width=width,
                height=height,
                ann={}
            )

            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            gt_polygons_ignore = []

            # parse all objects
            for obj in root.findall('object'):
                raw_cls = obj.find('name').text if obj.find('name') is not None else None
                if raw_cls is None:
                    continue
                cls = self._norm_class_name(raw_cls)
                if cls not in self.cat2label:
                    # 알 수 없는 클래스는 스킵 (필요하면 매핑에 추가)
                    continue
                label = self.cat2label[cls]

                polygon = None
                if self.xmltype == 'obb':
                    # 1) robndbox (x_left_top ...): 지원하면 여기에 파싱 추가
                    robnd = obj.find('robndbox')
                    if robnd is not None:
                        # 좌표가 left/right top/bottom로 주어지는 변종 포맷
                        try:
                            polygon = np.array([
                                float(robnd.find('x_left_top').text),
                                float(robnd.find('y_left_top').text),
                                float(robnd.find('x_right_top').text),
                                float(robnd.find('y_right_top').text),
                                float(robnd.find('x_right_bottom').text),
                                float(robnd.find('y_right_bottom').text),
                                float(robnd.find('x_left_bottom').text),
                                float(robnd.find('y_left_bottom').text),
                            ], dtype=np.float32)
                        except Exception:
                            polygon = None
                    # 2) polygon(x1..y4)
                    if polygon is None:
                        polygon = self._parse_polygon(obj)
                    # 3) fallback: bndbox -> polygon
                    if polygon is None:
                        bnd = obj.find('bndbox')
                        if bnd is not None:
                            polygon = self._hbb_to_poly(bnd)
                else:
                    # xmltype == 'hbb': bndbox만 사용
                    bnd = obj.find('bndbox')
                    if bnd is not None:
                        polygon = self._hbb_to_poly(bnd)

                if polygon is None:
                    continue

                # polygon -> obb (x_c, y_c, w, h, theta) by mmrotate core
                bbox = poly2obb_np(polygon, self.version)
                if bbox is None:
                    continue

                gt_bboxes.append(np.asarray(bbox, dtype=np.float32))
                gt_labels.append(label)
                gt_polygons.append(np.asarray(polygon, dtype=np.float32))

            # pack
            if gt_bboxes:
                data_info['ann']['bboxes'] = np.array(gt_bboxes, dtype=np.float32)
                data_info['ann']['labels'] = np.array(gt_labels, dtype=np.int64)
                data_info['ann']['polygons'] = np.array(gt_polygons, dtype=np.float32)
            else:
                data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
                data_info['ann']['labels'] = np.array([], dtype=np.int64)
                data_info['ann']['polygons'] = np.zeros((0, 8), dtype=np.float32)

            # ignore fields (not used here, but keep structure)
            if gt_polygons_ignore:
                data_info['ann']['bboxes_ignore'] = np.array(gt_bboxes_ignore, dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(gt_labels_ignore, dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.array(gt_polygons_ignore, dtype=np.float32)
            else:
                data_info['ann']['bboxes_ignore'] = np.zeros((0, 5), dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array([], dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.zeros((0, 8), dtype=np.float32)

            data_infos.append(data_info)

        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, info in enumerate(self.data_infos):
            if info['ann']['labels'].size > 0:
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
                 nproc=4):
        """Evaluate with rotated mAP."""
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
            for thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {thr}{"-" * 15}', logger=logger)
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        else:
            raise NotImplementedError

        return eval_results
