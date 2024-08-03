import os
import PIL
import numpy as np
import scipy.sparse
import torch

from torchvision.ops import box_iou

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')

class imdb(object):
    """Image database"""

    def __init__(self, name, classes=None):
        self._name = name
        self._num_classes = 0
        if not classes:
            self._classes = []
        else: 
            self._classes = classes
        self._image_index = []
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.default_roidb

        self.config = {}

    @property
    def name(self):
        return self._name
    
    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def image_index(self):
        return self._image_index
    
    @property
    def roidb_handler(self):
        return self._roidb_handler
    
    @roidb_handler.setter
    def roidb_handler(self, handler):
        self._roidb_handler = handler
    
    def set_proposal_method(self, method):
        method = eval('self.' + method + '_roidb')
        self.roidb_handler = method
    
    @property
    def roidb(self):
        """
        roidb is a list of dictionaries
        {
            'boxes',
            'gt_ious',
            'gt_classes',
            'flipped'
        }
        """
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def num_images(self):
        return len(self.image_index)
    
    def image_path_at(self, i):
        raise NotImplementedError
    
    def image_id_at(self, i):
        raise NotImplementedError
    
    def default_roidb(self):
        raise NotImplementedError
    
    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError
    
    def _get_widths(self):
        return [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in range(self.num_images)]
    
    def append_flipped_images(self):
        num_images= self.num_images
        widths = self._get_widths()
        for i in range(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] = oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            flipped_roi = {'boxes': boxes,
                           'gt_ious': self.roidb[i]['gt_ious'],
                           'gt_classes': self.roidb[i]['gt_classes'],
                           'flipped': True}
            self.roidb.append(flipped_roi)
        self._image_index = self._image_index * 2

    def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                        area='all', limit=None):
        """
        Evaluate detection proposal recall metrics

        Returns:
            results: dictionary of results with keys
            {
                'ar': average recall
                'recall': vector recalls at each IoU overlap threshold
                'thresholds': vector of IoU overlap thresholds
                'gt_overlaps': vector of all ground-truth overlaps
            }
        """
        areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
                 '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
        area_range = [
            [0 ** 2, 1e5 ** 2],  # all
            [0 ** 2, 32 ** 2],  # small
            [32 ** 2, 96 ** 2],  # medium
            [96 ** 2, 1e5 ** 2],  # large
            [96 ** 2, 128 ** 2],  # 96-128
            [128 ** 2, 256 ** 2],  # 128-256
            [256 ** 2, 512 ** 2],  # 256-512
            [512 ** 2, 1e5 ** 2],  # 512-inf
        ]
        assert area in areas, f'unknown area range: {area}'
        area_range = area_range[areas[area]]
        gt_ious = np.zeros(0)
        num_pos = 0

        for i in range(self.num_images):
            # Checking for max_overlaps == 1 avoids including crowd annotations
            max_gt_overlaps = self.roidb[i]['gt_ious'].toarray().max(axis=1)
            gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                               (max_gt_overlaps == 1))[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
            gt_areas = self.roidb[i]['seg_areas'][gt_inds]
            valid_gt_inds = np.where((gt_areas >= area_range[0]) & 
                                     (gt_areas <= area_range[1]))[0]
            gt_boxes = gt_boxes[valid_gt_inds, :]
            num_pos += len(valid_gt_inds)

            if candidate_boxes is None:
                non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
                boxes = self.roidb[i]['boxes'][non_gt_inds, :]
            else:
                boxes = candidate_boxes[i]

            if boxes.shape[0] == 0:
                continue
            
            if limit is not None and boxes.shape[0] > limit:
                boxes = boxes[:limit, :]

            ious = box_iou(torch.Tensor(boxes), torch.Tensor(gt_boxes)).numpy()

            _gt_ious = np.zeros((gt_boxes.shpae[0]))
            for j in range(gt_boxes.shape[0]):
                gt_argmax_ious = ious.argmax(axis = 0)
                gt_max_ious = ious.max(axis = 0)
                gt_ind = gt_max_ious.argmax()
                gt_iou = gt_max_ious.max()
                assert (gt_iou >= 0)
                box_ind = gt_argmax_ious[gt_ind]
                _gt_ious[j] = ious[box_ind, gt_ind]
                assert (_gt_ious[j] == gt_iou)
                # mark the proposal box and gt box as used
                ious[box_ind, :] = -1
                ious[:, gt_ind] = -1
            # append recorded iou coverage level
            gt_ious = np.hstack((gt_ious, _gt_ious))

        gt_ious = np.sort(gt_ious)
        if thresholds is None:
            step = 0.05
            thresholds = np.arange(0.5, 0.95 + 1e-5, step)

        recalls = np.zeros_like(thresholds)
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_ious >= t).sum() / float(num_pos)
        ar = recalls.mean()

        return {
            'ar': ar,
            'recalls': recalls,
            'thresholds': thresholds,
            'gt_ious': gt_ious
        }
    
    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
            'Number of boxes must math number of ground-truth images'
        roidb = []

        for i in range(self.num_iamges):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            ious = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_ious = box_iou(torch.Tensor(boxes), torch.Tensor(gt_boxes)).numpy()
                argmax_ious = gt_ious.argmax(axis=1)
                max_ious = gt_ious.max(axis=1)
                inds = np.where(max_ious > 0)[0]
                ious[inds, gt_classes[argmax_ious[inds]]] = max_ious[inds]

            ious = scipy.sparse.csr_matrix(ious)
            roidb.append({
                'boxes': boxes,
                'gt_classes': np.zeros((num_boxes, ) dtype=np.int32),
                'gt_ious': ious,
                'flipped': False,
                'seg_areas': np.zeros((num_boxes, ), dtype=np.float32)
            })
        
        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in range(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            a[i]['gt_ious'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                   b[i]['gt_overlaps']])
            a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                           b[i]['seg_areas']))
        return a