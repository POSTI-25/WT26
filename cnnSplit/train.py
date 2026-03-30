import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class SignLanguageDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)

        self.labels = data.iloc[:, 0].values
        self.images = data.iloc[:, 1:].values.astype(np.float32) / 255.0
        self.images = self.images.reshape(-1, 1, 28, 28)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx]),
            torch.tensor(self.labels[idx]).long()
        )


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 25)
        )

    def forward(self, x):
        return self.model(x)


def train():
    train_data = SignLanguageDataset("sign_mnist_train.csv")
    test_data = SignLanguageDataset("sign_mnist_test.csv")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {100*correct/total:.2f}%")

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()