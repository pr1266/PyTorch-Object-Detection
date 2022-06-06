import torch
from collections import Counter
from intersection_over_union import intersection_over_union
import numpy as np

def mean_average_precision(pred_boxes, true_boxes, iou_th=0.5, box_format='corners', num_classes=20):

    #! pred box list e mesle hint haye code haye ghabli
    average_precisions = []
    epsilon = 1e-6

    #! baraye har class bayad hesab konim:
    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
