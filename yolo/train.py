from cProfile import label
from xml.etree.ElementTree import PI
from pyrsistent import T
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

#! hyperparameters:
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

def Compose(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        
        return img, bboxes


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, level=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #! inja progress bar ro update mikonim:
        loop.set_postfix(loss=loss.item())
    print(f"Mean Loss : {sum(mean_loss)/len(mean_loss)}")


def main():

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    #! dataset ro dorost konim:
    train_dataset = VOCDataset(
        'data/8examples.csv',
        transform=transforms,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    test_dataset = VOCDataset(
        'data/test.csv',
        transform=transforms,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)

    for epoch in range(EPOCHS):
        
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.4, threshold=0.4
        )
        
        mean_avg_precision = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format='midpoint'
        )
        
        print(f"Train mAP: {mean_avg_precision}")

        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()