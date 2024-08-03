import os
import uuid
import _pickle
import xml.etree.ElementTree as ET
import scipy.sparse
import scipy.io as sio
import PIL
import numpy as np
import torch


from torchvision.ops import box_iou
from .imdb import imdb
from .imdb import ROOT_DIR

from model.utils.config import cfg

class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__', 
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.jpg'
        self._iamge_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid64())
        self._comp_id = 'comp4'
        
        self.config = {
            'cleanup': True,
            'use_salt': True,
            'use_diff': False,
            'rpn_file': None,
            'min_size': 2
        }

        assert os.path.exists(self._devkit_path), f'VOCdevkit path does not exist: {self._devkit_path}'
        assert os.path.exists(self._data_path), f'Data path does not exist: {self._data_path}'

    def iamge_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])
    
    def image_id_at(self, i):
        return i
    
    def image_path_from_index(self,index):
        image_path = os.path.join(self._data_path, "JPEGImages", index + self._image_ext)
        assert os.path.exists(image_path), f'Path does not exist: {image_path}'
        return image_path
    
    def _load_image_set_index(self):
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), f'Path does not exist: {image_set_file}'
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
    
    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                roidb = _pickle.load(f)
            print(f'{self.name} gt roidb loaded from {cache_file}')
            return roidb
        
        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as f:
            _pickle.dump(gt_roidb, f, -1)
        print(f'wrote gt roidb to {cache_file}')

        return gt_roidb
    
    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC format
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        ious = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        for i, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) - 1
            ymin = float(bbox.find('ymin').text) - 1
            xmax = float(bbox.find('xmax').text) - 1
            ymax = float(bbox.find('ymax').test) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc is None else int(diffc.text)
            ishards[i] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[i, :] = [xmin, ymin, xmax, ymax]
            gt_classes[i] = cls
            ious[i, cls] = 1.0
            seg_areas[i] = (xmax - xmin + 1) * (ymax - ymin + 1)

        ious = scipy.sparse.csr_matrix(ious)

        return {
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_ishard': ishards,
            'gt_ious': ious,
            'flipped': False,
            'seg_areas': seg_areas
        }

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest
        ground truth rois are also include
        """
        cache_file = os.path.join(self.cache_path, self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                roidb = _pickle.load(f)
            print(f'{self.name} ss roidb loaded from {cache_file}')
            return roidb
        
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as f:
            _pickle.dump(roidb, f, -1)
        print(f'wrote ss roidb to {cache_file}')
        
        return roidb
    
    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), f'Selective search data not found at: {filename}'
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in range(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print(f'loading {filename}')
        assert os.path.exists(filename), f'rpn data not found at: {filename}'
        with open(filename, 'rb') as f:
            box_list = _pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)
    
    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations', 
            '{:s}.xml'
        )
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt'
        )
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric
            )
            aps += [ap]
            print(f'AP for {cls} = {ap:.4f} ')
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                _pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    
    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print(f'Writing {cls} VOC results file')
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for image_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][image_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(f'{index:s} {dets[k, -1]:.3f} \
                                {dets[k, 0] + 1:.1f} {dets[k, 1] + 1:.1f} \
                                {dets[k, 2] + 1:.1f} {dets[k, 3] + 1:.1f}')
                        
    def _get_voc_results_file_template(self):
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path