import torch
from intersection_over_union import intersection_over_union

#! inja mikhaim behtarin bounding box ro peida konim:
#? hint: age iou bein e 2 ta bbox az threshold bishtar bashe,
#? farz mikonim ke 2 ta bbox daran be ye object eshare mikonan
#? pass ooni ke confidence bishtari dare ro negah midarim
#? aval e aval ke yek seri bbox haro ba threshold confidence
#? level filter mikonim bere asan, baadesh:
#TODO: take out the largest probablity box,
#TODO: and remove all other boxes with IoU > threshold
#TODO: do this for all classes!


def nms(bboxes, iou_threshold, threshold, box_format='corners'):
    
    #! prediction chie? prediction = [[class1, prob of class1, x1, y1, x2, y2], [same as first one], [same as first one]]
    assert type(bboxes) == list

    #TODO inja filter mikonim va ye seri ro bar asas e
    #TODO confidence level hazf mikonim:
    bboxes = [box for box in bboxes if box[1] > threshold]
    