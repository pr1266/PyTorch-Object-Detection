import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def intersection_over_union(boxes_preds, boxes_labels, box_format=''):
    #! inja box_preds 4 ta value dare: x1, y1, x2, y2
    #TODO inja baraye box1
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]

    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]

    #TODO inja baraye box2
    box2_x1 = boxes_preds[..., 0:1]
    box2_y1 = boxes_preds[..., 1:2]

    box2_x2 = boxes_preds[..., 2:3]
    box2_y2 = boxes_preds[..., 3:4]

    #! inja max haro dar miarim:
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
