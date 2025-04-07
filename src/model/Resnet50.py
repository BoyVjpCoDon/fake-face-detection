import torch
from torch import nn
import torchvision

def init_model_ResNet50(trainable_extractor = False, device = 'cuda'):
    """
    Create an ResNet50 model.

    Returns:
        model (torch.nn.Module): ResNet50 model.
    """
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).to(device)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = trainable_extractor
    
    model.fc= nn.Sequential(
    torch.nn.Linear(2048,1000),
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