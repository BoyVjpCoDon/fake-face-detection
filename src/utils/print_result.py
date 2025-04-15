import os
import torch

def print_results(checkpoint_dir, model_name):
    """
    Load results from checkpoints.

    Args:
        checkpoint_dir (str): Directory containing checkpoint files.

    Returns:
        results (dict): Dictionary containing lists of train_loss, train_acc, test_loss, and test_acc.
    """

    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and f.__contains__(model_name)])

    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint_path, f"{checkpoint['train_loss']:.4f}", f"{checkpoint['test_loss']:.4f}", f"{checkpoint['train_acc']:.4f}", f"{checkpoint['test_acc']:.4f}", sep=" | ")
if __name__ == "__main__":
    checkpoint_dir = "checkpoints"  # Directory containing checkpoint files
    model_name = "EfficientNetV2"  # Model name
    print("                                 name | train_loss | test_loss | tran_acc | test_acc ")
    print_results(checkpoint_dir, model_name)
