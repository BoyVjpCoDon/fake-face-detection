from torchvision import datasets
from torch.utils.data import DataLoader

def load_data(train_dir: str, valid_dir: str, batch_size: int = 64, transform=None):    # Load data
    train_data = datasets.ImageFolder(train_dir, transform=transform, target_transform = None)
    valid_data = datasets.ImageFolder(valid_dir, transform=transform)

    # Create data loaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, num_workers=1, shuffle=False)

    return train_dataloader, valid_dataloader