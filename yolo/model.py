import torch
import torch.nn as nn

"""
khob, bahal tarin bakhsh kar
tarahi model architecture
Information about architecture config:
in tuple ha intori an : (kernel_size, filters, stride, padding)
"M" laye maxpooling ba stride 2x2 va kernel 2x2 e
jaiiam ke list gozashtim tedad repeat e convLayer hast
"""
architecture_config = [
    #! kernel size, num_filters, stride and padding!
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CnnBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):

        super(CnnBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):

        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.leaky_relu(out)
        return out

