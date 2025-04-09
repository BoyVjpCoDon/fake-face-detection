import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

import sys
import os
src_path = (os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

from Resnet50_CA import resnet50_ca
# --- Coordinate Attention Module (giữ nguyên) ---
class Resnet50_MobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.spatial_stream = resnet50_ca()
        # weight = models.ResNet50_Weights.DEFAULT
        # self.spatial_stream.load_state_dict(weight.get_state_dict(progress=True), strict=False)
        # for name, param in self.spatial_stream.named_parameters():
        #     if name in weight.get_state_dict():
        #         param.requires_grad = False

        self.spatial_stream.fc = nn.Identity()

        self.frequency_stream = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.frequency_stream.classifier = nn.Identity()

        self.classify = nn.Sequential(
            nn.Linear(2048 + 1280, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(),
            nn.Linear(512,2)
        )
    
    def forward(self, x):
        # Spatial stream
        spatial_features = self.spatial_stream(x)

        # Frequency stream
        # Convert input to grayscale (average across channels)
        # grayscale_x = torch.mean(x, dim=1, keepdim=True)
        # Compute FFT
        fft_x = torch.fft.fft2(torch.view_as_complex(torch.stack([x, torch.zeros_like(x)], dim=-1)), dim=(-2, -1))
        # Compute magnitude
        magnitude_spectrum = torch.abs(fft_x)
        # Normalize magnitude (optional, but can be helpful)
        batch_size = x.size(0)
        min_val = torch.min(magnitude_spectrum.view(batch_size, -1), dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_val = torch.max(magnitude_spectrum.view(batch_size, -1), dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        magnitude_spectrum_normalized = (magnitude_spectrum - min_val) / (max_val - min_val + 1e-8)

        # Resize magnitude to 224x224 (MobileNetV2 input size)
        frequency_input = torch.nn.functional.interpolate(magnitude_spectrum_normalized, size=(224, 224), mode='bilinear', align_corners=False)

        frequency_features = self.frequency_stream(frequency_input)

        # Combine features
        combined_features = torch.cat((spatial_features, frequency_features), dim=1)

        # Fully connected layers
        output = self.classify(combined_features)

        return output

if __name__ == "__main__":
    model = Resnet50_MobileNetV2()
    from torchinfo import summary
    summary(model, depth=3, input_size=(16,3,224,224), col_names=["input_size","output_size", "kernel_size", 'mult_adds', 'trainable'], row_settings=["var_names"])