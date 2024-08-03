import numpy as np
import numpy.random as npr
import cv2

from model.utils.config import cfg
from model.utils.blob import *

def get_minibatch(roidb, num_classes):
    num_images = len(roidb)
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        f'num_images ({num_images}) must divide BATCH_SIZE ({cfg.TRAIN.BATCH_SIZE})'
    
    image_blob, image_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': image_blob}

    assert len(image_scales) == 1, 'Single batch only'
    assert len(roidb) == 1, 'Single batch only'

    # gt boxes: (xmin, ymin, xmax, ymax, cls)
    if cfg.TRAIN.USE_ALL_GT:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0 &
                           np.all(roidb[0]['gt_ious'].toarray() > -1.0), axis=1)[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * image_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['image_info'] = np.array(
        [[image_blob.shape[1], image_blob.shape[2], image_scales[0]]],
        dtype=np.float32
    )

    blobs['image_id'] = roidb[0]['image_id']

    return blobs

def _get_image_blob(roidb, scale_inds):
    """
    Builds an input blob from the images in the roidb at the specified scales
    """
    num_images = len(roidb)

    processed_images = []
    image_scales = []

    for i in range(num_images):
        image = cv2.imread(roidb[i]['image'])

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.concatenate((image, image, image), axis=2)
        
        # image channel bgr -> rgb
        image = image[:, :, ::-1]

        if roidb[i]['flipped']:
            image = image[:, ::-1, :]  
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        image, image_scale = prepare_image_for_blob(image, cfg.PIXEL_MEANS,
                                                    target_size,
                                                    cfg.TRAIN.MAX_SIZE)
        image_scales.append(image_scale)
        processed_images.append(image)
    
    image_blob = image_list_to_blob(processed_images)

    return image_blob, image_scales