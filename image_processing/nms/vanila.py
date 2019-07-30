import numpy as np


def box_non_max_suppression(boxes, threshold=0.3):
    """
    处理在同一张图片上大量彼此重叠的方框区域的非极值抑制算法。
    Args:
        boxes: 彼此重叠的方框列表。输入的方框存储格式为(x, y, width, height)。
        threshold: 两个方框重叠面积与当前方框面积之比大于此阈值时会将当前的方框抑制。

    Returns: 经过抑制过滤后的方框区域列表。

    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the area of the bounding box
    area = boxes[:, 2] * boxes[:, 3]
    indexes = np.argsort(area)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(indexes) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last_index = len(indexes) - 1
        target_index = indexes[last_index]
        pick.append(target_index)
        suppress = [last_index]
        # loop over all indexes in the indexes list
        for pos in range(0, last_index):
            # grab the current index
            current_index = indexes[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[target_index], x1[current_index])
            yy1 = max(y1[target_index], y1[current_index])
            xx2 = min(x2[target_index], x2[current_index])
            yy2 = min(y2[target_index], y2[current_index])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[current_index]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > threshold:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        indexes = np.delete(indexes, suppress)

        # return only the bounding boxes that were picked
    return boxes[pick]