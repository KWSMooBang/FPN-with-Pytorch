import numpy as np

def unique_boxes(boxes, scale=1.0):
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)

def xywh_to_xyxy(boxes):
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

def validate_boxes(boxes, width=0, height=0):
  xmin = boxes[:, 0]
  ymin = boxes[:, 1]
  xmax = boxes[:, 2]
  ymax = boxes[:, 3]
  assert (xmin >= 0).all()
  assert (ymin >= 0).all()
  assert (xmax >= xmin).all()
  assert (ymax >= ymin).all()
  assert (xmax < width).all()
  assert (ymax < height).all()


def filter_small_boxes(boxes, min_size):
  w = boxes[:, 2] - boxes[:, 0]
  h = boxes[:, 3] - boxes[:, 1]
  keep = np.where((w >= min_size) & (h > min_size))[0]
  return keep