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

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.arc = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.arc)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        out = torch.flatten(x, start_dim=1)
        out = self.fcs(out)
        return out

    def _create_conv_layers(self, arc):
        layers = []
        in_channels = self.in_channels

        for x in arc:
            if type(x) == tuple:
                layers += [
                    CnnBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                #! inja repeatation darim:
                conv1 = x[0] #! in tuple e 
                conv2 = x[1] #! inam hamintor
                num_repeats = x[2] #! in integer e ke tedad repeatation e
                for _ in range(num_repeats):
                    layers += [
                        CnnBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CnnBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    #! inja ye hint bedam
                    #! har seri ke in 2 ta convLayer posht ham mian
                    #! input channel e har seri output channel e seri qable
                    #! normale! chera? chon output shape har convLayer hamoon
                    #! input shape convLayer baddie, pas:
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            # nn.Linear(1024*S*S, 4096)
            nn.Linear(1024*S*S, 496), #! too original paper ino 4096 gozashte
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(C+B*5)), # (S, S, 30)
        )

def test(S=7, B=2, C=20):
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)

test()