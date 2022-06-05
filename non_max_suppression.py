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
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms