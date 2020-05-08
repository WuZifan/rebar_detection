
import numpy as np
import torch

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2

        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

if __name__ == '__main__':

    detections = np.load('detection.npy')
    print(detections.shape)
    temp_cnt = len(detections)

    detections = torch.from_numpy(detections)

    nms_thres = 0.4
    keep_boxes = []

    while_cnt = 0

    flag = False
    while detections.size(0):
        '''
            如果先找label_match;然后在label_match里面的结果找IOU_Match;
            那么由于IOU_match处理的结果不是全部数据，那么就比较难过滤了

            所以最好是先全量IOU，然后再找label
        '''
        while_cnt += 1

        if while_cnt >= temp_cnt + 4:
            flag=True
        # 用第一张图片和后面每一张图片做IOU计算，并判断哪些位置满足IOU需求
        # 这样的话自己一定会被干掉
        print('{} boxes left,{} in total'.format(len(detections), temp_cnt))

        # 处理nan and inf
        temp = np.array(np.isnan(detections[0]),dtype=np.int8)
        if np.sum(temp)>0 or np.inf in detections[0]:
            print('there are nan or inf in detection[0],drop')
            detections = np.delete(detections,0,axis=0)
            continue

        large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
        if flag:
            print('large_overlap',large_overlap)

        # 找到和自己label相同的框
        label_match = detections[0, -1] == detections[:, -1]

        # 找到和自己IOU过大，并且label相同的元素
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match

        if flag:
            print('invalid: ',invalid)

        # 把所有去掉的框用加权的方式来获得当前框
        weights = detections[invalid, 4:5]
        # Merge overlapping bboxes by order of confidence
        detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()

        if flag:
            print('dete',detections)
            for a in detections:
                temp = np.array(np.isnan(a),dtype=np.int8)
                if np.sum(temp)>0:
                    print('nan: ',a)
                if np.inf in a:
                    print('inf ',a)

        # 保存当前框
        keep_boxes += [detections[0]]

        # 拿到剩下的框
        detections = detections[~invalid]
        if flag:
            break


