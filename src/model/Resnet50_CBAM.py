import torch
from torch import nn
import torchvision


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class BottleneckWithCBAM(torchvision.models.resnet.Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckWithCBAM, self).__init__(inplanes, planes, stride, downsample)
        self.cbam = CBAM(planes * self.expansion)
        self.planes = planes

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def init_model_ResNet50_CBAM(trainable_extractor = False, device='cuda'):
    """
    Create an ResNet50 model with CBAM attention.

    Returns:
        model (torch.nn.Module): ResNet50 model with CBAM.
    """
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).to(device)

    for name, module in model.named_children():
        if name not in ['conv1', 'bn1', 'relu', 'maxpool', 'fc']:
            for block_name, block in module.named_children():
                if isinstance(block, torchvision.models.resnet.Bottleneck):
                    inplanes = block.conv1.in_channels
                    planes = block.conv2.out_channels # Thử lấy planes từ conv2
                    stride = block.conv2.stride[0]
                    downsample = block.downsample
                    setattr(module, block_name, BottleneckWithCBAM(inplanes, planes, stride, downsample))

    # Freeze layers (tương tự như hàm gốc)
    for param in model.parameters():
        param.requires_grad = trainable_extractor

    # Thay thế lớp fully connected
    model.fc= nn.Sequential(
        torch.nn.Linear(2048,1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024,512),
        torch.nn.Dropout(),
        torch.nn.Linear(in_features=512,
                        out_features=2,
                        bias=True)
    )
    # Chuyển toàn bộ model lên device sau khi đã thực hiện các thay đổi
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # model.fc = model.fc.to(device) # Đảm bảo cả lớp fc cũng được chuyển

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    return model