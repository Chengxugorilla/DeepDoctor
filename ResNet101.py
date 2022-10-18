import torch.nn as nn
from torch.nn import functional as F



class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        out = self.down(x)
        return out


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.pre = nn.Sequential(nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )
        self.conv1_first = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, padding=0, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False)
                                         )
        self.conv1_next = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, padding=0, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 256, kernel_size=1, padding=0, bias=False),
                                        nn.BatchNorm2d(256)
                                        )

        self.conv2_first = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, padding=0, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False),
                                         nn.BatchNorm2d(512)
                                         )
        self.conv2_next = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, padding=0, bias=False),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False),
                                        nn.BatchNorm2d(512)
                                        )

        self.conv3_first = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 1024, kernel_size=1, padding=0, bias=False),
                                         nn.BatchNorm2d(1024)
                                         )
        self.conv3_next = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, padding=0, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 1024, kernel_size=1, padding=0, bias=False),
                                        nn.BatchNorm2d(1024)
                                        )

        self.conv4_first = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(512, 2048, kernel_size=1, padding=0, bias=False),
                                         nn.BatchNorm2d(2048),
                                         )
        self.conv4_next = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1, padding=0, bias=False),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 2048, kernel_size=1, padding=0, bias=False),
                                        nn.BatchNorm2d(2048),
                                        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(nn.Dropout(p=0.5),
                                nn.Linear(2048, 1000),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(1000, 1)
                                )

        self.layer1_shortcut = DownSample(64, 256, 1)
        self.layer2_shortcut = DownSample(256, 512, 1)
        self.layer3_shortcut = DownSample(512, 1024, 1)
        self.layer4_shortcut = DownSample(1024, 2048, 1)


    def forward(self, x):
        out = self.pre(x)
        layer1_identity = self.layer1_shortcut(out)
        out = self.conv1_first(out)
        out = F.relu(out + layer1_identity, inplace=True)

        for i in range(2):
            identity = out
            out = self.conv1_next(out)
            out = F.relu(out + identity, inplace=True)

        layer2_identity = self.layer2_shortcut(out)
        out = self.conv2_first(out)
        out = F.relu(out + layer2_identity, inplace=True)

        for i in range(3):
            identity = out
            out = self.conv2_next(out)
            out = F.relu(out + identity, inplace=True)

        layer3_identity = self.layer3_shortcut(out)
        out = self.conv3_first(out)
        out = F.relu(out + layer3_identity, inplace=True)

        for i in range(22):
            identity = out
            out = self.conv3_next(out)
            out = F.relu(out + identity, inplace=True)

        layer4_identity = self.layer4_shortcut(out)
        out = self.conv4_first(out)
        out = F.relu(out + layer4_identity, inplace=True)

        for i in range(2):
            identity = out
            out = self.conv4_next(out)
            out = F.relu(out + identity, inplace=True)

        out = self.avg_pool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out.squeeze(-1)



