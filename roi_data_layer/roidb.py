import PIL
import numpy as np
import datasets

from model.utils.config import cfg
from datasets.factory import get_imdb

def prepare_roidb(imdb):
    roidb = imdb.roidb

    if not (imdb.name.startswith('coco')):
        sizes = [PIL.Image.open(imdb.image_path_at(i)).size
                 for i in range(imdb.num_images)]
        
    for i in range(len(imdb.image_index)):
        roidb[i]['image_id'] = imdb.image_id_at(i)
        roidb[i]['image'] = imdb.image_path_at(i)
        if not (imdb.name.startswith('coco')):
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
        
        gt_ious = roidb[i]['gt_ious'].toarray()
        max_ious = gt_ious.max(axis=1)
        max_classes = gt_ious.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_ious'] = max_ious
        zero_inds = np.where(max_ious == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        nonzero_inds = np.where(max_ious > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

def rank_roidb_ratio(roidb):
    ratio_large = 2
    ratio_small = 0.5

    ratio_list= []
    for i in range(len(roidb)):
        width = roidb[i]['width']
        height = roidb[i]['height']
        ratio = width / float(height)

        if ratio > ratio_large:
            roidb[i]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]['need_crop'] = 1
            ratio = ratio_small
        else:
            roidb[i]['need_crop'] = 0

        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    print(f'before filtering, there are {len(roidb)} images...')
    i = 0
    while i < len(roidb):
        if len(roidb[i]['boxes'] == 0):
            del roidb[i]
            i -=1
        i += 1

    print(f'after filtering, there are {len(roidb)} images...')
    return roidb

def combined_roidb(imdb_names, training=True):
    """
    Combine multiple roidbs
    """
    def get_training_roidb(imdb):
        if cfg.TRAIN.USE_FLIPPED:
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()
            print('done')
        
        print('Preparing training data...')

        prepare_roidb(imdb)
        print('done')

        return imdb.roidb
    
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print(f'Loaded dataset `{imdb.name:s}` for training')
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print(f'Set proposal method: {cfg.TRAIN.PROPOSAL_METHOD:s}')
        roidb = get_training_roidb(imdb)
        return roidb
    
    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]

    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)

    if training:
        roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    return imdb, roidb, ratio_list, ratio_index