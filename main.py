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

epochs_clean = 3      # nb d epoques pour le modele base
epochs_adv = 3        # nb d epoques pour le modele adversarial

epsilon = 0.25        # intensite de la perturbation FGSM
alpha_adv = 0.5       # poids clean vs adversarial dans la loss

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
# 2. Modele simple
# =========================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # entree: 1 x 28 x 28
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
# 3. Entrainement et test clean
# =========================

def train_one_epoch_clean(model, loader, optimizer, epoch, device):
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
    print(f"[Train clean] Epoch {epoch} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")


def test_model_clean(model, loader, device, label="Model"):
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
    print(f"[{label}] Test clean - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    return avg_loss, acc

# =========================
# 4. FGSM - Fast Gradient Sign Method
# =========================

def fgsm_attack(model, images, labels, epsilon, device):
    """
    FGSM: x_adv = x + epsilon * sign(grad_x J(theta, x, y))
    """
    model.eval()

    images = images.clone().detach().to(device)
    images.requires_grad = True
    labels = labels.to(device)

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    grad_sign = images.grad.data.sign()
    adv_images = images + epsilon * grad_sign
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()

# =========================
# 5. Test sur adversarial examples
# =========================

def test_model_adversarial(model, loader, epsilon, device, label="Model"):
    model.eval()
    adv_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        adv_data = fgsm_attack(model, data, target, epsilon, device)

        outputs = model(adv_data)
        loss = F.cross_entropy(outputs, target, reduction="sum")
        adv_loss += loss.item()

        pred = outputs.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = adv_loss / total
    acc = correct / total
    print(f"[{label}] Test adversarial eps={epsilon} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    return avg_loss, acc

# =========================
# 6. Entrainement adversarial (adversarial training)
# =========================

def train_one_epoch_adversarial(model, loader, optimizer, epoch, device, epsilon, alpha):
    """
    Adversarial training:
    J_tilde = alpha * J(x, y) + (1 - alpha) * J(x_adv, y)
    avec x_adv genere par FGSM.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        # generation des adversarial examples pour ce batch
        adv_data = fgsm_attack(model, data, target, epsilon, device)

        optimizer.zero_grad()

        output_clean = model(data)
        output_adv = model(adv_data)

        loss_clean = F.cross_entropy(output_clean, target)
        loss_adv = F.cross_entropy(output_adv, target)

        loss = alpha * loss_clean + (1 - alpha) * loss_adv
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)

        pred = output_clean.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"[Train adv] Epoch {epoch} - Loss: {avg_loss:.4f}, Clean Acc: {acc:.4f}")

# =========================
# 7. Main - comparaison des deux modeles
# =========================

def main():
    # ----- Modele de base, entraine en clean -----
    print("\n=== Entrainement du modele de base (clean) ===")
    base_model = SimpleCNN().to(device)
    optimizer_base = optim.Adam(base_model.parameters(), lr=lr)

    for epoch in range(1, epochs_clean + 1):
        train_one_epoch_clean(base_model, train_loader, optimizer_base, epoch, device)
        test_model_clean(base_model, test_loader, device, label="Base model")

    print("\nEvaluation du modele de base sur adversarial examples:")
    base_clean_loss, base_clean_acc = test_model_clean(base_model, test_loader, device, label="Base model")
    base_adv_loss, base_adv_acc = test_model_adversarial(base_model, test_loader, epsilon, device, label="Base model")

    # ----- Modele avec adversarial training -----
    print("\n=== Entrainement du modele adversarial (adversarial training) ===")
    adv_model = SimpleCNN().to(device)
    optimizer_adv = optim.Adam(adv_model.parameters(), lr=lr)

    for epoch in range(1, epochs_adv + 1):
        train_one_epoch_adversarial(adv_model, train_loader, optimizer_adv, epoch, device, epsilon, alpha_adv)
        test_model_clean(adv_model, test_loader, device, label="Adv model")

    print("\nEvaluation du modele adversarial sur adversarial examples:")
    adv_clean_loss, adv_clean_acc = test_model_clean(adv_model, test_loader, device, label="Adv model")
    adv_adv_loss, adv_adv_acc = test_model_adversarial(adv_model, test_loader, epsilon, device, label="Adv model")

    # ----- Resume comparatif -----
    print("\n=== Resume comparatif ===")
    print(f"Modele de base - Clean Acc: {base_clean_acc:.4f}, Adv Acc: {base_adv_acc:.4f}")
    print(f"Modele adv    - Clean Acc: {adv_clean_acc:.4f}, Adv Acc: {adv_adv_acc:.4f}")

if __name__ == "__main__":
    main()
