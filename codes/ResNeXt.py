import torch
import torch.nn as nn
import torch.nn.functional as F

# Default: BN before ReLU(dosen't make sense)
# next try : pre-activation (BN->ReLU->Conv)

class ResidualXtBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=256, stride=1, cardinality=16, width=64):
        
        super(ResidualXtBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Short cut
        residual = self.shortcut(x) if hasattr(self, 'shortcut') else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = F.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, ResidualBlock, cardinality, width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.width = width
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, bias=False, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(ResidualXtBlock, in_channels=64, width=64,  out_channels=256,  num_blocks=3, stride=2)
        self.layer2 = self.make_layer(ResidualXtBlock, in_channels=256, width=128, out_channels=512,  num_blocks=3, stride=2) 
        self.layer3 = self.make_layer(ResidualXtBlock, in_channels=512, width=256, out_channels=512,  num_blocks=3, stride=2) 
        self.layer4 = self.make_layer(ResidualXtBlock, in_channels=512, width=256, out_channels=512,  num_blocks=3, stride=2) 
        self.layer5 = self.make_layer(ResidualXtBlock, in_channels=512, width=256, out_channels=1024,  num_blocks=3, stride=1)
        self.fc = nn.Linear(1024, num_classes)


    def make_layer(self, block, in_channels, width, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #if num_blocks = 2, stride=2, strides = [2,1] (reduce dim at first block)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, stride, self.cardinality, width))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))       # (3, 256, 256)   -> (64, 128, 128)
        out = self.layer1(out)                      # (64, 128, 128  -> (256, 64, 64)
        out = self.layer2(out)                      # (256, 64, 64) -> (512, 32, 32)
        out = self.layer3(out)                      # (512, 32, 32) -> (1024, 16, 16)
        out = self.layer4(out)                      # (512, 16, 16) -> (1024, 8, 8)
        out = self.layer5(out)                      
        out = F.avg_pool2d(out, kernel_size=(8,8))  # (1024, 8, 8)  -> (1024, 1, 1)      
        out = out.view(out.size(0), -1)             # (1024, 1, 1)  -> (1024, 1)
        out = self.fc(out)                          # (1024, 1)     -> (340, 1)
        return out


def ResNeXt29_4x64d(num_classes=340):
    return ResNeXt(ResidualXtBlock, cardinality=32, width=64, num_classes=num_classes)

def ResNet34():
    return ResNet(ResidualBlock, num_blocks_list=[3,4,6,3])

def ResNet50():
    return ResNet(ResidualBlock, num_blocks_list=[3,4,6,3])

def ResNet101():
    return ResNet(ResidualBlock, num_blocks_list=[3,4,23,3])

def ResNet152():
    return ResNet(ResidualBlock, num_blocks_list=[3,8,36,3])


def test_ResNeXt():
    net = ResNeXt29_4x64d()
    print(net)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(pytorch_total_params)
    x = torch.randn(1,3,256,256)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test_ResNeXt()









