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

        #! avalin parameter pred train_idx e
        #! tooye khat zir, ye dictionary darim az in ke img key chand
        #! ta bbox dare tooye valuesh
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        #! inja confusion matrix ro tashkil midim:
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_boxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )