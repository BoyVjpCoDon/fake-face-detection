import sys
import os
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_path)
from utils.train import train
import torch
import os
from dataloader.dataloader import load_data
from model.Resnet50_MobileNetV2 import Resnet50_MobileNetV2
from torchvision.models import ResNet50_Weights

if __name__ == "__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    train_dir="archive/dataset/train"
    valid_dir="archive/dataset/valid"
    test_dir="archive/dataset/test"

    model = Resnet50_MobileNetV2().to(device)
    train_dataloader, test_dataloader = load_data(train_dir,
                                                   valid_dir,
                                                   batch_size  = 32,
                                                   transform=ResNet50_Weights.DEFAULT.transforms())

    train(model=model,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
          checkpoint_model_name="Resnet50_MobileNetV2",
          epochs=15,
          pretrained="")