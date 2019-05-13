import torch
import torch.nn as nn
import torch.nn.functional as F


num_country = 10
# Try BN after ReLU

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_1x1, n_3x3red, n_3x3, n_5x5red, n_5x5, n_pool):
        super(InceptionBlock, self).__init__()
        # 1x1 conv branch
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, n_1x1, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(n_1x1),  
        )

        # 1x1 conv -> 3x3 conv branch
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, n_3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(n_3x3red),
            nn.Conv2d(n_3x3red, n_3x3, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(n_3x3),
        )

        # 1x1 conv -> 5x5 conv branch
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, n_5x5red, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(n_5x5red),
            nn.Conv2d(n_5x5red, n_5x5, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(n_5x5),
            nn.Conv2d(n_5x5, n_5x5, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(n_5x5),
        )

        # 3x3 pool -> 1x1 conv branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, n_pool, kernel_size=1),
            nn.BatchNorm2d(n_pool),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.branch_1x1(x)
        y2 = self.branch_3x3(x)
        y3 = self.branch_5x5(x)
        y4 = self.branch_pool(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=340):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(1, 192, kernel_size=3, padding=1, stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(192),
        )
        # (in_channels, n_1x1, n_3x3red, n_3x3, n_5x5red, n_5x5, n_pool)
        self.a3 = InceptionBlock(192,  64,  96, 128, 16, 32, 32) # 1x1 64, 3x3 128, 5x5 32, pool 32
        self.b3 = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = InceptionBlock(480, 192,  96, 208, 16,  48,  64)
        self.b4 = InceptionBlock(512, 160, 112, 224, 24,  64,  64)
        self.c4 = InceptionBlock(512, 128, 128, 256, 24,  64,  64)
        self.d4 = InceptionBlock(512, 112, 144, 288, 32,  64,  64)
        self.e4 = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.a5 = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024+num_country, num_classes)

    def forward(self, x, country):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat([out, country], 1)
        out = self.linear(out)
        return out

def test_GoogLeNet():
    net = GoogLeNet()
    x = torch.randn(1,1,64,64)
    country = torch.randn(1,10)
    y = net(x, country)
    print(y.size())

if __name__ == '__main__':
    test_GoogLeNet()