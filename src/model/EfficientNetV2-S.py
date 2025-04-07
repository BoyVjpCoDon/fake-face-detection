import torch
from torch import nn
import torchvision

def init_model_efficientnet_v2_s(trainable_extractor = False, device = 'cuda'):
    """
    Create an EfficientNetV2-S model.

    Returns:
        model (torch.nn.Module): EfficientNetV2-S model.
    """
    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_s(weights=weights).to(device)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = trainable_extractor
    
    model.classifier= nn.Sequential(
    torch.nn.Linear(1280,1000),
    torch.nn.ReLU(),
    torch.nn.Linear(1000,500),
    torch.nn.Dropout(),
    torch.nn.Linear(in_features=500,
                    out_features=2,
                    bias=True)
    ).to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Recreate the classifier layer and seed it to the target device
    
    return model 