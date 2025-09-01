# utils/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import pandas as pd
from utils.preprocessing import preprocess_image

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)  # Assumes CSV with 'image_path', 'label' columns
        self.root_dir = root_dir
        self.classes = ["pictures", "screenshots", "documents"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['image_path'])
        image = cv2.imread(img_path)
        image = preprocess_image(image)
        label = self.classes.index(self.data.iloc[idx]['label'])
        return torch.from_numpy(image).float().permute(2, 0, 1), label

class OCRClassifier(nn.Module):
    def __init__(self):
        super(OCRClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_class = nn.Linear(32 * 56 * 56, 3)  # Classification head
        self.fc_ocr = nn.Linear(32 * 56 * 56, 10)  # Dummy OCR head; adjust for real OCR

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        class_logits = self.fc_class(x)
        ocr_logits = self.fc_ocr(x)
        return class_logits, ocr_logits

def train_model():
    dataset = ImageDataset("dataset/labels.csv", "dataset/")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = OCRClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    class_criterion = nn.CrossEntropyLoss()
    ocr_criterion = nn.CrossEntropyLoss()  # Adjust for OCR task

    for epoch in range(10):
        for images, labels in loader:
            optimizer.zero_grad()
            class_logits, ocr_logits = model(images)
            class_loss = class_criterion(class_logits, labels)
            loss = class_loss  # Add ocr_loss when implemented
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), "model/custom_ocr_model.pt")

if __name__ == "__main__":
    train_model()
