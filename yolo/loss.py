import torch
import torch.nn as nn
from utils import intersection_over_union

"""
khob bahs e zibaye loss
loss e yolo yekam dardesar dare
avalan yolo miad 2 ta box dar nazar migire
yeki amoodi yeki ofoghi
baad tashkhis mide kodoom behtare
oonon migire anchor box
az tarafi, har cell faghat mitoone yek class ro
present kone, hala harchi resolution grid bishtar bashe
tedad class ha ro mitoonim bebarim bala tar dar natije tedad
object ha ham bishtar mishe
hala loss esh chetor hesab mishe?
vaghti ye box predict mikonim, 2 ta halat ro dar nazar migirim
1 - aya object dare?
2 - aya khalie?
output marboot be har cell, ye vector dare
ke tooch class confidence dare
age object nabashe oonja, confidence ro 0 mizare
hala too oon 2 ta bounding box e age object too har 2 tash
bashe chi? oonio negah midarim ke IoU ba ground truthesh bishtare
hala chand ta loss darim?
pesar ajab loss function ghashangi dare
aval ma miaim x, y ro ke center e bbox hast predict mikonim
pas ye parameter e loss vase x, y
baad miaim mibinim ke width o height e in bounding box
cheghad error dare, pas shod 2 ta ta inja
baad miaim 2 ta score ii ke baraye vojood ya adam e vojood e
object dar bbox hastan ro optimize mikonim va dar nahayat, confidence level ro
ke dar vaghe harchi in ehtemal be 1 nazdik tar bashe ba etminan e bishtari darim
predict ro anjam midim
"""
class YoloLoss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C

        #! 2 ta lambda darim vase koochik kardan e loss
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        #! 2 ta bounding box darim har kodoom 5 ta parameter:
        #! x, y, w, h, c
        #! 20 ta ham tedad class hamoone (too in dataset)
        #! va baraye har cell ya split 1 doone mikhaim
        #! pass predictions shape esh N * S * S tast ke har kodoom az ina 30 ta element daran
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        #! hala ground truth chi? oon ye bounding box dare
        #! pass size esh mishe 25 ta
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        #! inja behtarinesho entekhab mikonim
        iou_maxes, bestbox = torch.max(ious, dim=0)

        #! in element e 21 om chie?
        #! goftim ke ye confidence level darim
        #! ke age object oonja nabashe, 0 mishe
        #! dar gheir e in soorat ye adad bein e 0 o 1 mishe
        exists_box = target[..., 20].unsqueeze(3)  

        #! in bestbox e ya 0 e ya 1 ke indexesh mishe dige
        #! va ba in formool oon yeki box hazf mishe
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        #! in 2 ta command e baadi ro nafahmidam chera estefade karde
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        #! khob hala biaim loss hesab konim:
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        #! va hame loss haro jam mikonim ba ham:
        loss = (
            self.lambda_coord * box_loss  
            + object_loss  
            + self.lambda_noobj * no_object_loss  
            + class_loss  
        )

        return loss
