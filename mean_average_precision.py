import torch
from collections import Counter
from intersection_over_union import intersection_over_union
import numpy as np

def mean_average_precision(pred_boxes, true_boxes, iou_th=0.5, box_format='corners', num_classes=20):

    