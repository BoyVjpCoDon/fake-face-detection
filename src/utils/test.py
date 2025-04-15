import torch
from tqdm import tqdm
from torch import nn

import os
import sys
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)

import torchvision.models as models
from dataloader.dataloader import load_data
from utils.load_checkpoint import load_checkpoint
from model.Resnet50 import init_model_ResNet50
from model.EfficientNetV2_S import init_model_efficientnet_v2_s
from model.ViT_B_16 import init_model_Transformer
from model.Resnet50_CBAM import init_model_ResNet50_CBAM
from model.Resnet50_CA import resnet50_ca
from model.Resnet50_LKA import resnet50_lka

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def test_model(model, checkpoint_path, test_loader, device = "cuda"):
    """
    Đánh giá mô hình phân loại trên tập dữ liệu.

    Args:
        model (torch.nn.Module): Mô hình PyTorch cần đánh giá.
        checkpoint_path (str): Đường dẫn đến file checkpoint của mô hình.
        dataloader (torch.utils.data.DataLoader): DataLoader cho tập dữ liệu test.
        device (str): Thiết bị để chạy mô hình ('cuda' hoặc 'cpu').

    Returns:
        dict: Dictionary chứa các metric đánh giá (accuracy, f1-score, precision, recall, classification_report).
    """
    model.to(device)
    model.load_state_dict(load_checkpoint(checkpoint_path)[0])
    model.eval()  # Chuyển mô hình sang chế độ đánh giá
    all_preds = []
    all_labels = []
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for (X, y) in tqdm(test_loader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()


            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            all_labels.extend(y.cpu())
            all_preds.extend(test_pred_labels.cpu())
            

    # Adjust metrics to get average loss per batch
    test_loss = test_loss / len(test_loader)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    report = classification_report(all_labels, all_preds)

    results = {
        "loss": loss,
        "accuracy": accuracy,
        "f1-score": f1,
        "precision": precision,
        "recall": recall,
        "classification_report": report
    }

    print("\n--- Kết quả đánh giá ---")
    print(f"loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:\n", report)

    return results

if __name__ == '__main__':
    # Tạo một checkpoint giả định
    test_path = "archive/dataset/test"
    #sua transform theo model
    _, test_loader = load_data(test_path, test_path, batch_size=64, transform=models.EfficientNet_V2_S_Weights.DEFAULT.transforms())
    #sua checkpoint
    checkpoint_path = "checkpoints/EfficientNetV2_S_epoch_24.pth"
    #sua model
    model = init_model_efficientnet_v2_s()

    # Gọi hàm test
    test_model(model, checkpoint_path, test_loader)