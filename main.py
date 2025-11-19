import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# 0. Config de base
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 128
test_batch_size = 1000
lr = 1e-3
epochs = 3

# =========================
# 1. Chargement de MNIST
# =========================

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


def describe_dataset(train_dataset, test_dataset):
    print("Taille train :", len(train_dataset))
    print("Taille test  :", len(test_dataset))
    print("Nombre de classes :", len(train_dataset.classes))
    print("Classes :", train_dataset.classes)

describe_dataset(train_dataset, test_dataset)

# =========================
# 2. Modèle simple
# =========================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # entrée: 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 32 x 28 x 28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 x 28 x 28
        self.pool = nn.MaxPool2d(2, 2)                            # 64 x 14 x 14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =========================
# 3. Fonctions train et test (clean)
# =========================

def train_one_epoch(model, loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"[Train] Epoch {epoch} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")


def test_model(model, loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction="sum")
            test_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    avg_loss = test_loss / total
    acc = correct / total
    print(f"[Test clean] Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    return avg_loss, acc

# =========================
# 4. Main
# =========================

def main():
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, epoch, device)
        test_model(model, test_loader, device)

if __name__ == "__main__":
    main()
