import torch
from torch import nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

def init_model_ResNet50(trainable_extractor = False):
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

def load_checkpoint(checkpoint_path):
    """
    Load model and optimizer state from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        model (torch.nn.Module): The model with loaded state.
        optimizer (torch.optim.Optimizer): The optimizer with loaded state.
        epoch (int): The epoch at which the checkpoint was saved.
    """
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    optimizer = checkpoint['optimizer_state_dict']
    epoch = checkpoint['epoch']
    return model_state_dict, optimizer, epoch

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for (X, y) in tqdm(dataloader, desc="Batch"):
        # Send data to target device
        # print("\rbatch: " + str(batch) + "/" + str(round(int(100000/64))), end = "")
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          checkpoint_model_name: str = "",
          epochs: int = 5,
          pretrained: str = None):
    # 1. Take in various parameters required for training and test steps

    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    if pretrained:
        model_state_dict, optimizer_state_dict, start_epoch = load_checkpoint(pretrained)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        start_epoch = 0
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in range(start_epoch+1, start_epoch + epochs):
        print("Epoch:",epoch)
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # 6. Save Checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        torch.save(checkpoint, f"checkpoints/{checkpoint_model_name}_epoch_{epoch}.pth")
        
    # 7. Return the filled results at the end of the epochs
    return results

def load_data(train_dir: str, valid_dir: str, batch_size: int = 64):
    # Define transforms
    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    auto_transforms = weights.transforms()

    # Load data
    train_data = datasets.ImageFolder(train_dir, transform=auto_transforms, target_transform = None)
    valid_data = datasets.ImageFolder(valid_dir, transform=auto_transforms)

    # Create data loaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, num_workers=1, shuffle=False)

    return train_dataloader, valid_dataloader

if __name__ == "__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    train_dir="archive/dataset/train"
    valid_dir="archive/dataset/valid"
    test_dir="archive/dataset/test"

    model = init_model_ResNet50()
    # Train
    batch_size = 64
    train_dataloader, test_dataloader = load_data(train_dir, valid_dir, batch_size)

    train(model=model,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
          checkpoint_model_name="ResNet50",
          epochs=10,
          pretrained="")
