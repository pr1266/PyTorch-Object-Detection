import torch
import torch.nn as nn

#! inja config network ro dorost mikonim:
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 
            'C', 512, 512, 512, 'M', 512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}