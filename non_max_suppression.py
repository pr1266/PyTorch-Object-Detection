
#! inja mikhaim behtarin bounding box ro peida konim:
#? hint: age iou bein e 2 ta bbox az threshold bishtar bashe,
#? farz mikonim ke 2 ta bbox daran be ye object eshare mikonan
#? pass ooni ke confidence bishtari dare ro negah midarim
import torch
from intersection_over_union import intersection_over_union

def nms(bboxes, iou_threshold, threshold, box_format='corners'):
    pass