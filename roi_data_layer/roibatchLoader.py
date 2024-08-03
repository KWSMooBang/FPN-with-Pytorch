import time
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image
from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch
from model.rpn.bbox_utils import bbox_transform_inv, clip_boxes

class roibatchLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size,
                 num_classes, training=True, normalize=None):
        self._roidb = roidb
        self._num_classes = num_classes
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.TRIM_WIDTH
        self.max_num_box = cfg.MAX_NUM_GT_BOXES
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)

        self.ratio_list_batch = torch.zeros(self.data_size)
        num_batch = int(np.ceil(len(ratio_index) / batch_size))
        for i in range(num_batch):
            left_ind =  i * batch_size
            right_ind = min((i + 1) * batch_size-1, self.data_size-1)

            if ratio_list[right_ind] < 1:
                target_ratio = ratio_list[left_ind]
            elif ratio_list[left_ind] > 1:
                target_ratio = ratio_list[right_ind]
            else:
                target_ratio = 1

            self.ratio_list_batch[left_ind: (right_ind+1)] = target_ratio

    def __getitem__(self, index):
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        minibatch_db = [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db, self._num_classes)
        data = torch.from_numpy(blobs['data'])
        image_info = torch.from_numpy(blobs['image_info'])

        data_height, data_width = data.size(1), data.size(2)
        if self.training:
            np.random.shuffle(blobs['gt_boxes'])
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])

            ########################################################
            # padding the input image to fixed size for each group #
            ########################################################

            # NOTE1: need to cope with the case where a group cover both conditions. (done)
            # NOTE2: need to consider the situation for the tail samples. (no worry)
            # NOTE3: need to implement a parallel data loader. (no worry)
            # get the index range

            # if the image need to crop, crop to the target size.
            ratio = self.ratio_list_batch[index]

            if self._roidb[index_ratio]['need_crop']:
                if ratio < 1:
                    # this means that data_width << data_height, we need to crop the
                    # data_height
                    min_y = int(torch.min(gt_boxes[:,1]))
                    max_y = int(torch.max(gt_boxes[:,3]))
                    trim_size = int(np.floor(data_width / ratio))
                    box_region = max_y - min_y + 1
                    if min_y == 0:
                        y_s = 0
                    else:
                        if (box_region-trim_size) < 0:
                            y_s_min = max(max_y-trim_size, 0)
                            y_s_max = min(min_y, data_height-trim_size)
                            if y_s_min == y_s_max:
                                y_s = y_s_min
                            else:
                                y_s = np.random.choice(range(y_s_min, y_s_max))
                        else:
                            y_s_add = int((box_region-trim_size)/2)
                            if y_s_add == 0:
                                y_s = min_y
                            else:
                                y_s = np.random.choice(range(min_y, min_y+y_s_add))
                    # crop the image
                    data = data[:, y_s:(y_s + trim_size), :, :]

                    # shift y coordiante of gt_boxes
                    gt_boxes[:, 1] = gt_boxes[:, 1] - y_s
                    gt_boxes[:, 3] = gt_boxes[:, 3] - y_s

                    # update gt bounding box according the trip
                    gt_boxes[:, 1].clamp_(0, trim_size - 1)
                    gt_boxes[:, 3].clamp_(0, trim_size - 1)

                else:
                    # this means that data_width >> data_height, we need to crop the
                    # data_width
                    min_x = int(torch.min(gt_boxes[:,0]))
                    max_x = int(torch.max(gt_boxes[:,2]))
                    trim_size = int(np.ceil(data_height * ratio))
                    box_region = max_x - min_x + 1
                    if min_x == 0:
                        x_s = 0
                    else:
                        if (box_region-trim_size) < 0:
                            x_s_min = max(max_x-trim_size, 0)
                            x_s_max = min(min_x, data_width-trim_size)
                            if x_s_min == x_s_max:
                                x_s = x_s_min
                            else:
                                x_s = np.random.choice(range(x_s_min, x_s_max))
                        else:
                            x_s_add = int((box_region-trim_size)/2)
                            if x_s_add == 0:
                                x_s = min_x
                            else:
                                x_s = np.random.choice(range(min_x, min_x+x_s_add))
                    # crop the image
                    data = data[:, :, x_s:(x_s + trim_size), :]

                    # shift x coordiante of gt_boxes
                    gt_boxes[:, 0] = gt_boxes[:, 0] - x_s
                    gt_boxes[:, 2] = gt_boxes[:, 2] - x_s
                    # update gt bounding box according the trip
                    gt_boxes[:, 0].clamp_(0, trim_size - 1)
                    gt_boxes[:, 2].clamp_(0, trim_size - 1)

            # based on the ratio, padding the image.
            if ratio < 1:
                # this means that data_width < data_height
                trim_size = int(np.floor(data_width / ratio))

                padding_data = torch.zeros(int(np.ceil(data_width / ratio)), \
                                                data_width, 3, dtype=torch.float32)

                padding_data[:data_height, :, :] = data[0]
                # update im_info
                im_info[0, 0] = padding_data.size(0)
                # print("height %d %d \n" %(index, anchor_idx))
            elif ratio > 1:
                # this means that data_width > data_height
                # if the image need to crop.
                padding_data = torch.zeros(data_height, \
                                                int(np.ceil(data_height * ratio)), 3, dtype=torch.float32)
                padding_data[:, :data_width, :] = data[0]
                im_info[0, 1] = padding_data.size(1)
            else:
                trim_size = min(data_height, data_width)
                padding_data = torch.zeros(trim_size, trim_size, 3, dtype=torch.float32)
                padding_data = data[0][:trim_size, :trim_size, :]
                gt_boxes.clamp_(0, trim_size)
                im_info[0, 0] = trim_size
                im_info[0, 1] = trim_size


            # check the bounding box:
            not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
            keep = torch.nonzero(not_keep == 0).view(-1)

            gt_boxes_padding = torch.zeros(self.max_num_box, gt_boxes.size(1), dtype=torch.float32)
            if keep.numel() != 0:
                gt_boxes = gt_boxes[keep]
                num_boxes = min(gt_boxes.size(0), self.max_num_box)
                gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
            else:
                num_boxes = 0

                # permute trim_data to adapt to downstream processing
            padding_data = padding_data.permute(2, 0, 1).contiguous()
            im_info = im_info.view(3)

            return padding_data, im_info, gt_boxes_padding, num_boxes
        else:
            data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
            im_info = im_info.view(3)

            gt_boxes = torch.FloatTensor([1,1,1,1,1])
            num_boxes = 0

            return data, im_info, gt_boxes, num_boxes

    def __len__(self):
        return len(self._roidb)

