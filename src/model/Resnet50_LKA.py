import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck
import torchvision

class LargeKernelAttn(nn.Module):
    def __init__(self, channels):
        super(LargeKernelAttn, self).__init__()
        self.dwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            groups=channels
        )
        self.dwdconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            padding=9,
            groups=channels,
            dilation=3
        )
        self.pwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1
        )

    def forward(self, x):
        weight = self.pwconv(self.dwdconv(self.dwconv(x)))

        return x * weight

class BottleneckWithLKA(Bottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ca = LargeKernelAttn(self.conv3.out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def resnet50_lka(**kwargs):
    model = ResNet(block=BottleneckWithLKA, layers=[3, 4, 6, 3])
    model.fc= nn.Sequential(
        torch.nn.Linear(2048,1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024,512),
        torch.nn.Dropout(),
        torch.nn.Linear(in_features=512,
                        out_features=2,
                        bias=True)
    )

    weight = torchvision.models.ResNet50_Weights.DEFAULT
    model.load_state_dict(weight.get_state_dict(progress=True), strict=False)
    
    for name, param in model.named_parameters():
        if name in weight.get_state_dict():
            param.requires_grad = False
    model.to(device="cuda" if torch.cuda.is_available() else "cpu")
    return model

from torchinfo import summary
print(summary(resnet50_lka(), input_size=(32,3, 224, 224), col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds", "trainable"], row_settings=["var_names"], depth=4))
