import torch
from torch import nn
import torchvision

def init_model_Transformer(trainable_extractor = False, device = 'cuda'):
    """
    Create a Vision Transformer (ViT) model.

    Args:
        num_classes (int): Number of classes for the final fully connected layer.
        device (torch.device): The device to run the model on.

    Returns:
        model (torch.nn.Module): Vision Transformer model.
    """
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights).to(device)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = trainable_extractor
    
    model.heads = nn.Sequential(
        nn.Linear(model.heads.head.in_features, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 2, bias=True)
    ).to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Recreate the classifier layer and seed it to the target device
    
    return model 