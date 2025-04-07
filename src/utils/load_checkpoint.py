import torch

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