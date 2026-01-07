# multi_dataset_protocol.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score # étude performance pure
import matplotlib.pyplot as plt
import time

# =========================
# Configuration utilisateur
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# valeurs à ajuster pour CPU
batch_size = 128
test_batch_size = 1000
lr = 1e-3

# Par défaut mis bas pour exécution raisonnable sur CPU
epochs_clean = 1
epochs_adv = 1

epsilon = 0.25
alpha_adv = 0.5
n_visual = 5  # nb d'exemples à visualiser par dataset

# Pour accélérer les tests sur CPU, tu peux commenter CIFAR ci-dessous
datasets_info = {
    "MNIST": {
        "dataset": datasets.MNIST,
        "channels": 1,
        "transform": transforms.ToTensor(),
        "classes": 10
    },
    "Fashion-MNIST": {
        "dataset": datasets.FashionMNIST,
        "channels": 1,
        "transform": transforms.ToTensor(),
        "classes": 10
    },
    "CIFAR-10": {
        "dataset": datasets.CIFAR10,
        "channels": 3,
        "transform": transforms.Compose([transforms.ToTensor()]),
        "classes": 10
    }
}

# =========================
# Modèle CNN adaptable
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(         # dropout+batchnorm, amélioration itérative
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # calcul de la taille du feature map après pooling (28->14, 32->16)
        if in_channels == 1:
            feat_size = 14  # pour MNIST-like 28x28
        else:
            feat_size = 16  # pour CIFAR-like 32x32
        self.fc1 = nn.Linear(64 * feat_size * feat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =========================
# Fonctions d'entraînement / test
# =========================
def train_one_epoch_clean(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        running_correct += pred.eq(target).sum().item()
        running_total += data.size(0)
    return running_loss / running_total, running_correct / running_total

# accuracy → performance globale et F1-score → plus robuste aux classes difficiles
def test_model_clean(model, loader, device, return_f1=False):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            correct += pred.eq(target).sum().item()
            total += data.size(0)

            if return_f1:
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

    acc = correct / total

    if return_f1:
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        return acc, f1

    return acc

# =========================
# FGSM (corrigé)
# =========================
def fgsm_attack(model, images, labels, epsilon, device):
    """
    images : tensor (B,C,H,W)
    returns adversarial images (B,C,H,W)
    """
    model.eval()
    images = images.to(device)
    # on s'assure d'avoir un tenseur qui require grad
    images = images.clone().detach().requires_grad_(True)
    labels = labels.to(device)

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    # grad peut être None si quelque chose s'est mal passé ; on vérifie
    if images.grad is None:
        raise RuntimeError("images.grad is None after backward() in fgsm_attack")

    adv_images = images + epsilon * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0.0, 1.0)
    return adv_images.detach()

# si le modèle résiste à pgd (attk forte) => robuste
def pgd_attack(model, images, labels, epsilon, alpha, num_iter, device):
    """
    PGD attack (l_inf)
    images : tensor (B,C,H,W)
    """
    model.eval()

    images = images.to(device)
    labels = labels.to(device)

    # point de départ = image originale + petit bruit
    adv_images = images.clone().detach()
    adv_images = adv_images + 0.001 * torch.randn_like(adv_images)
    adv_images = torch.clamp(adv_images, 0, 1)

    for _ in range(num_iter):
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        # étape de gradient
        grad_sign = adv_images.grad.sign()
        adv_images = adv_images + alpha * grad_sign

        # projection dans la boule epsilon
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + eta, 0, 1).detach()

    return adv_images


def test_model_adversarial(model, loader, epsilon, device):
    model.eval()
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        # on GENERE adv examples (besoin de grad) -> ne pas entourer par torch.no_grad()
        adv_data = fgsm_attack(model, data, target, epsilon, device)
        # on prédit ensuite sans grad
        with torch.no_grad():
            output = model(adv_data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return correct / total

def test_model_adversarial_pgd(model, loader, epsilon, alpha, num_iter, device):
    model.eval()
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        adv_data = pgd_attack(
            model,
            data,
            target,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            device=device
        )

        with torch.no_grad():
            output = model(adv_data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    return correct / total

# ajout test bruit gaussien non adversarial => fiabilité
def test_model_noise(model, loader, sigma, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            noisy_data = data + torch.randn_like(data) * sigma
            noisy_data = torch.clamp(noisy_data, 0, 1)

            output = model(noisy_data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    return correct / total

def train_one_epoch_adversarial(model, loader, optimizer, epsilon, alpha, device, print_every=100):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        # génération adv pour ce batch (coûteux)
        adv_data = fgsm_attack(model, data, target, epsilon, device)

        optimizer.zero_grad()
        output_clean = model(data)
        output_adv = model(adv_data)
        loss_clean = F.cross_entropy(output_clean, target)
        loss_adv = F.cross_entropy(output_adv, target)
        loss = alpha * loss_clean + (1.0 - alpha) * loss_adv
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        pred = output_clean.argmax(dim=1)
        running_correct += pred.eq(target).sum().item()
        running_total += data.size(0)

        if (batch_idx + 1) % print_every == 0:
            print(f"  [adv train] batch {batch_idx+1}/{len(loader)}")
    return running_loss / running_total, running_correct / running_total

# ==============
# Visualisation 
# ==============
def show_adversarial_examples(model, loader, epsilon, device, n=5):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:n].to(device), labels[:n].to(device)
    adv_images = fgsm_attack(model, images, labels, epsilon, device)

    # Ramener CPU pour affichage
    images = images.cpu().detach()
    adv_images = adv_images.cpu().detach()

    fig = plt.figure(figsize=(3*n, 6))
    for i in range(n):
        # Original
        ax1 = fig.add_subplot(3, n, i+1)
        img = images[i].permute(1, 2, 0).squeeze()
        cmap = "gray" if img.ndim == 2 else None
        ax1.imshow(img, cmap=cmap)
        ax1.set_title(f"Orig: {labels[i].item()}")
        ax1.axis("off")

        # Adv
        ax2 = fig.add_subplot(3, n, n+i+1)
        img2 = adv_images[i].permute(1, 2, 0).squeeze()
        cmap2 = "gray" if img2.ndim == 2 else None
        ax2.imshow(img2, cmap=cmap2)
        ax2.set_title("Adv")
        ax2.axis("off")

        # Différence
        ax3 = fig.add_subplot(3, n, 2*n+i+1)
        diff = adv_images[i] - images[i]
        diff = diff.permute(1, 2, 0).squeeze()
        if diff.ndim == 3:
            diff_to_show = diff[..., 0]  # rendre lisible si RGB
        else:
            diff_to_show = diff
        ax3.imshow(diff_to_show, cmap="seismic", vmin=-epsilon, vmax=epsilon)
        ax3.set_title("Diff")
        ax3.axis("off")

    plt.tight_layout()
    plt.show()  # <-- maintenant bloquant

def show_pgd_examples(model, loader, epsilon, alpha, num_iter, device, n=5):
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images[:n].to(device), labels[:n].to(device)

    adv_images = pgd_attack(
        model,
        images,
        labels,
        epsilon,
        alpha,
        num_iter,
        device
    )

    images = images.cpu()
    adv_images = adv_images.cpu()

    fig = plt.figure(figsize=(3*n, 6))

    for i in range(n):
        # ---------- ORIGINAL ----------
        ax1 = fig.add_subplot(3, n, i+1)
        img = images[i]

        if img.shape[0] == 1:  # MNIST-like
            ax1.imshow(img.squeeze(), cmap="gray")
        else:  # CIFAR-like
            ax1.imshow(img.permute(1, 2, 0))

        ax1.set_title("Orig")
        ax1.axis("off")

        # ---------- PGD ----------
        ax2 = fig.add_subplot(3, n, n+i+1)
        adv = adv_images[i]

        if adv.shape[0] == 1:
            ax2.imshow(adv.squeeze(), cmap="gray")
        else:
            ax2.imshow(adv.permute(1, 2, 0))

        ax2.set_title("PGD")
        ax2.axis("off")

        # ---------- DIFF ----------
        ax3 = fig.add_subplot(3, n, 2*n+i+1)
        diff = adv_images[i] - images[i]

        if diff.shape[0] == 1:
            ax3.imshow(diff.squeeze(), cmap="seismic",
                       vmin=-epsilon, vmax=epsilon)
        else:
            # afficher une seule composante pour lisibilité
            ax3.imshow(diff[0], cmap="seismic",
                       vmin=-epsilon, vmax=epsilon)

        ax3.set_title("Diff")
        ax3.axis("off")

    plt.tight_layout()
    plt.show()

# =========================
# Trace et comparatif
# =========================
def plot_comparison(results):
    labels = list(results.keys())

    clean_acc = [results[l]['clean'] for l in labels]
    fgsm_acc = [results[l]['fgsm'] for l in labels]
    pgd_acc  = [results[l]['pgd']  for l in labels]

    x = range(len(labels))
    width = 0.25

    plt.figure(figsize=(9,5))
    plt.bar(x, clean_acc, width=width, label='Clean')
    plt.bar([i + width for i in x], fgsm_acc, width=width, label='FGSM')
    plt.bar([i + 2*width for i in x], pgd_acc, width=width, label='PGD')

    plt.xticks([i + width for i in x], labels)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Robustesse multi-dataset")
    plt.legend()
    plt.show()

# =========================
# Main : traitement dataset par dataset
# =========================
def main():
    overall_results = {}
    for name, info in datasets_info.items():
        print(f"\n=== Dataset: {name} ===")
        dataset_cls = info["dataset"]
        transform = info["transform"]
        in_channels = info["channels"]
        num_classes = info["classes"]

        # loaders
        train_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform)
        test_dataset = dataset_cls(root="./data", train=False, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        # modèle pour ce dataset
        model = SimpleCNN(in_channels, num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # --- Clean training ---
        print(" Training clean...")
        for epoch in range(1, epochs_clean + 1):
            t0 = time.time()
            loss, acc = train_one_epoch_clean(model, train_loader, optimizer, device)
            t1 = time.time()
            print(f"  Epoch {epoch}/{epochs_clean} - loss: {loss:.4f}, acc: {acc:.4f} (took {t1-t0:.1f}s)")

        clean_acc, clean_f1 = test_model_clean(model, test_loader, device, return_f1=True)
        # test adv (génération d'adv examples; ne pas entourer de no_grad)
        adv_acc = test_model_adversarial(model, test_loader, epsilon, device)
        print(f" Results (clean model) -> Clean Acc: {clean_acc:.4f}, Adv Acc: {adv_acc:.4f}")

        # Visualisation courte
        try:
            show_adversarial_examples(model, test_loader, epsilon, device, n=min(n_visual, len(test_dataset)))
        except Exception as e:
            print("  Visualisation failed:", e)

        # --- Adversarial training (séparé) ---
        print(" Training adversarial...")
        model_adv = SimpleCNN(in_channels, num_classes).to(device)
        optimizer_adv = optim.Adam(model_adv.parameters(), lr=lr)
        for epoch in range(1, epochs_adv + 1):
            t0 = time.time()
            loss_a, acc_a = train_one_epoch_adversarial(model_adv, train_loader, optimizer_adv, epsilon, alpha_adv, device)
            t1 = time.time()
            print(f"  Epoch {epoch}/{epochs_adv} - loss: {loss_a:.4f}, clean-acc: {acc_a:.4f} (took {t1-t0:.1f}s)")

        adv_clean_acc, adv_clean_f1 = test_model_clean(
            model_adv, test_loader, device, return_f1=True
        )
        adv_adv_acc = test_model_adversarial(
            model_adv, test_loader, epsilon, device
        )
        print(f" Results (adv model) -> Clean Acc: {adv_clean_acc:.4f}, Adv Acc: {adv_adv_acc:.4f}")

        pgd_acc = test_model_adversarial_pgd(
            model,
            test_loader,
            epsilon=epsilon,
            alpha=epsilon / 10,
            num_iter=10,
            device=device
        )

        pgd_adv_acc = test_model_adversarial_pgd(
            model_adv,
            test_loader,
            epsilon=epsilon,
            alpha=epsilon / 10,
            num_iter=10,
            device=device
        )
        
        noise_acc = test_model_noise(model, test_loader, sigma=0.1, device=device)

        show_pgd_examples(
            model,
            test_loader,
            epsilon,
            alpha=epsilon / 10,
            num_iter=10,
            device=device,
            n=n_visual
        )

        print(
            f" Results (clean model) -> "
            f"Clean Acc: {clean_acc:.4f}, "
            f"FGSM Acc: {adv_acc:.4f}, "
            f"PGD Acc: {pgd_acc:.4f}"
        )

        print(
            f" Results (adv model) -> "
            f"Clean Acc: {adv_clean_acc:.4f}, "
            f"FGSM Acc: {adv_adv_acc:.4f}, "
            f"PGD Acc: {pgd_adv_acc:.4f}"
        )

        overall_results[name] = {
            'clean': clean_acc,
            'f1': clean_f1,
            'fgsm': adv_acc,
            'pgd': pgd_acc,
            'noise': noise_acc,
            'adv_clean': adv_clean_acc,
            'adv_fgsm': adv_adv_acc,
            'adv_pgd': pgd_adv_acc
        }

    # Comparatif global
    print("\n=== Comparatif global ===")
    plot_comparison({k: v for k, v in overall_results.items()})

if __name__ == "__main__":
    main()
