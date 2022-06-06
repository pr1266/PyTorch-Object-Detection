import torch
from collections import Counter
from intersection_over_union import intersection_over_union
"""
khob ye hint molaii biam baratoon
tooye mean average precision aval miaim
hame bounding box 'predictions' ro rooye test set be dast miarim
baad ground truth haro ham darim dige label zadim
baad tooye in bounding box ha miaim TP va FP ro hesab mikonim
in chetor anjam mishe?
ahaaaaaaaaa
miaim iou ground truth va prediction box ro bedast miarim
age iou bishtar az threshol bashe FP tashkhis midim
yani:
1 - ya kolan object nadarim
2 - bounding boxemoon dorost predict nashode 
va masalan nesf e object ro dare localize mikone
vaghti TP va FP ha moshakhas shod, miaim va graph
precision-recall ro rasm mikonim va sath zir e nemoodar ro
hesab mikonim dar vaghea integral migirim
va baraye hame class ha inkaro mikonim va average migirim
ta score mean average precision be dast biad
"""
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
        #TODO arzam be khedmatetoon ke
        #TODO ta injaye kar rooye class haye mokhtalef
        #TODO iterate kardim va detection ha va GT haro daravordim

        #! avalin parameter pred train_idx e
        #! tooye khat zir, ye dictionary darim az in ke img key chand
        #! ta bbox dare tooye valuesh
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        #! inja detection haro bar asas e confidence level sort mikonim
        detections.sort(key=lambda x: x[2], reverse=True)
        #! inja confusion matrix ro tashkil midim:
        #! 2 ta tensor zero tashkil midim baadan process mikonim
        #! ke kodoom element ha bayad 1 bashan        
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truth_img)
            best_iou = 0
            #! hala behtarin anchor box ro dar miarim:
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            #! inja check mikonim ke bar asas e iou threshold
            #! bbox predict shode TP hast ya na
            if best_iou > iou_th:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        #! sath e zir e nemoodar precision-recall ro integral migirim:
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
                    