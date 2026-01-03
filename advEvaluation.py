# Reprise du code des 2A
# Ajout de visualisation
# Ajout de PGD

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt # visualisation

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

pgd_alpha = epsilon/20      # pas de GD pixel par pixel
pgd_steps = 20        # nombre d'itérations PGD

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
# 4bis. PGD - Descente de gradient pixel par pixel
# =========================

def pgd_attack(model, images, labels, epsilon, alpha, num_iter, device):
    """
    PGD (Projected Gradient Descent)
    Descente de gradient pixel par pixel
    """
    model.eval()

    images = images.clone().detach().to(device)
    labels = labels.to(device)

    # Initialisation aléatoire dans la boule epsilon
    adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1)

    for _ in range(num_iter):
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Mise à jour pixel par pixel
        grad_sign = adv_images.grad.data.sign()
        adv_images = adv_images + alpha * grad_sign

        # Projection dans la boule epsilon
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + eta, 0, 1).detach()

    return adv_images

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

def test_model_adversarial_pgd(model, loader, epsilon, alpha, num_iter, device, label="Model"):
    model.eval()
    adv_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        adv_data = pgd_attack(model, data, target, epsilon, alpha, num_iter, device)

        outputs = model(adv_data)
        loss = F.cross_entropy(outputs, target, reduction="sum")
        adv_loss += loss.item()

        pred = outputs.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = adv_loss / total
    acc = correct / total
    print(f"[{label}] PGD eps={epsilon} - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
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

# Visu
def show_adversarial_examples(model, loader, epsilon, device, n=5):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    adv_images = fgsm_attack(model, images, labels, epsilon, device)

    images = images.cpu().detach()
    adv_images = adv_images.cpu().detach()

    plt.figure(figsize=(12, 6))
    for i in range(n):
        # Original
        plt.subplot(3, n, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Orig: {labels[i].item()}")
        plt.axis('off')

        # Adversarial
        plt.subplot(3, n, n + i + 1)
        plt.imshow(adv_images[i].squeeze(), cmap='gray')
        plt.title("Adv")
        plt.axis('off')

        # Différence
        plt.subplot(3, n, 2*n + i + 1)
        diff = (adv_images[i] - images[i]).squeeze()
        plt.imshow(diff, cmap='seismic', vmin=-epsilon, vmax=epsilon)
        plt.title("Diff")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def show_pgd_examples(model, loader, epsilon, alpha, steps, device, n=5):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:n].to(device), labels[:n].to(device)

    adv_images = pgd_attack(model, images, labels, epsilon, alpha, steps, device)

    images = images.cpu()
    adv_images = adv_images.cpu()

    plt.figure(figsize=(12, 6))
    for i in range(n):
        # original
        plt.subplot(3, n, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Orig: {labels[i].item()}")
        plt.axis('off')

        # PGD
        plt.subplot(3, n, n + i + 1)
        plt.imshow(adv_images[i].squeeze(), cmap='gray')
        plt.title("PGD")
        plt.axis('off')

        # différence
        plt.subplot(3, n, 2*n + i + 1)
        diff = (adv_images[i] - images[i]).squeeze()
        plt.imshow(diff, cmap='seismic', vmin=-epsilon, vmax=epsilon)
        plt.title("Diff")
        plt.axis('off')

    plt.suptitle(f"PGD attack (ε={epsilon}, steps={steps})")
    plt.tight_layout()
    plt.show()

def plot_comparison(
    base_clean, base_fgsm, base_pgd,
    adv_clean, adv_fgsm, adv_pgd
):
    labels = ['Base model', 'Adv model']
    clean = [base_clean, adv_clean]
    fgsm  = [base_fgsm, adv_fgsm]
    pgd   = [base_pgd, adv_pgd]
    
    x = range(len(labels))
    width = 0.25

    plt.figure(figsize=(9,5))
    plt.bar(x, clean, width, label='Clean')
    plt.bar([i + width for i in x], fgsm, width, label='FGSM')
    plt.bar([i + 2*width for i in x], pgd, width, label='PGD')

    plt.xticks([i + width for i in x], labels)
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("Robustesse face aux attaques adversariales")
    plt.legend()
    plt.show()
    
# =========================
# 7. Main - comparaison des deux modeles
# =========================

def main():
    # MODELE DE BASE (clean)
    print("\n=== Entrainement du modele de base (clean) ===")
    base_model = SimpleCNN().to(device)
    optimizer_base = optim.Adam(base_model.parameters(), lr=lr)

    for epoch in range(1, epochs_clean + 1):
        train_one_epoch_clean(base_model, train_loader, optimizer_base, epoch, device)
        test_model_clean(base_model, test_loader, device, label="Base model")

    print("\n=== Evaluation du modele de base ===")

    # ---- Clean ----
    base_clean_loss, base_clean_acc = test_model_clean(
        base_model, test_loader, device, label="Base model"
    )

    # ---- FGSM ----
    base_fgsm_loss, base_fgsm_acc = test_model_adversarial(
        base_model, test_loader, epsilon, device, label="Base model (FGSM)"
    )

    # ---- PGD ----
    base_pgd_loss, base_pgd_acc = test_model_adversarial_pgd(
        base_model, test_loader, epsilon, pgd_alpha, pgd_steps, device, label="Base model (PGD)"
    )

    print("\nVisualisation FGSM (modele de base)")
    show_adversarial_examples(base_model, test_loader, epsilon, device, n=5)

    print("\nVisualisation PGD (modele de base)")
    show_pgd_examples(base_model, test_loader, epsilon, pgd_alpha, pgd_steps, device)

    # MODELE ADVERSARIAL
    print("\n=== Entrainement du modele adversarial ===")
    adv_model = SimpleCNN().to(device)
    optimizer_adv = optim.Adam(adv_model.parameters(), lr=lr)

    for epoch in range(1, epochs_adv + 1):
        train_one_epoch_adversarial(
            adv_model, train_loader, optimizer_adv,
            epoch, device, epsilon, alpha_adv
        )
        test_model_clean(adv_model, test_loader, device, label="Adv model")

    print("\n=== Evaluation du modele adversarial ===")

    # ---- Clean ----
    adv_clean_loss, adv_clean_acc = test_model_clean(
        adv_model, test_loader, device, label="Adv model"
    )

    # ---- FGSM ----
    adv_fgsm_loss, adv_fgsm_acc = test_model_adversarial(
        adv_model, test_loader, epsilon, device, label="Adv model (FGSM)"
    )

    # ---- PGD ----
    adv_pgd_loss, adv_pgd_acc = test_model_adversarial_pgd(
        adv_model, test_loader, epsilon, pgd_alpha, pgd_steps, device, label="Adv model (PGD)"
    )

    print("\nVisualisation FGSM (modele adversarial)")
    show_adversarial_examples(adv_model, test_loader, epsilon, device, n=5)

    print("\nVisualisation PGD (modele adversarial)")
    show_pgd_examples(adv_model, test_loader, epsilon, pgd_alpha, pgd_steps, device)

    # Résumé final
    print("\n=== Resume comparatif ===")
    print(f"Base model - Clean Acc : {base_clean_acc:.4f}")
    print(f"Base model - FGSM Acc  : {base_fgsm_acc:.4f}")
    print(f"Base model - PGD Acc   : {base_pgd_acc:.4f}")

    print(f"Adv model  - Clean Acc : {adv_clean_acc:.4f}")
    print(f"Adv model  - FGSM Acc  : {adv_fgsm_acc:.4f}")
    print(f"Adv model  - PGD Acc   : {adv_pgd_acc:.4f}")

    plot_comparison(
        base_clean_acc, base_fgsm_acc, base_pgd_acc,
        adv_clean_acc,  adv_fgsm_acc,  adv_pgd_acc
)
if __name__ == "__main__":
    main()