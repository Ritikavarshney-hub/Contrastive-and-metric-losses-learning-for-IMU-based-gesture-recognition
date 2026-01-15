import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Dataset ----------------
class EMGDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.y = np.argmax(np.load(y_path), axis=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

# ---------------- CNN Encoder ----------------
class CNNEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=10),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(64, 256, kernel_size=10),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=10),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, emb_dim)  # map CNN output to embedding

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, channels, seq_len)
        x = self.net(x).squeeze(-1)  # (B, 256)
        z = F.normalize(self.fc(x), dim=1)
        return z

# ---------------- Full Model ----------------
class FullModel(nn.Module):
    def __init__(self, emb_dim=128, num_classes=62):
        super().__init__()
        self.encoder = CNNEncoder(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return z, logits

# ---------------- Margin Embedding Loss ----------------
class MarginEmbeddingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z, y):
        dist = torch.cdist(z, z, p=2)
        y = y.unsqueeze(1)
        pos_mask = (y == y.T).float()
        neg_mask = (y != y.T).float()
        eye = torch.eye(len(z), device=z.device)
        pos_mask = pos_mask - eye
        pos_loss = pos_mask * dist.pow(2)
        neg_loss = neg_mask * F.relu(self.margin - dist).pow(2)
        loss = (pos_loss.sum() + neg_loss.sum()) / (pos_mask.sum() + neg_mask.sum() + 1e-8)
        return loss

# ---------------- Early Stopping ----------------
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_acc = -1
        self.counter = 0
        self.best_state = None

    def step(self, acc, model):
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore(self, model):
        model.load_state_dict(self.best_state)

# ---------------- Training & Evaluation ----------------
def train_and_eval(model, train_loader, test_loader, margin_loss, alpha=1.0, epochs=50, patience=10):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=5e-4)
    stopper = EarlyStopping(patience)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            z, logits = model(x)
            loss = ce_loss(logits, y) + alpha * margin_loss(z, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                _, logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1:02d} | Test Acc: {acc:.4f}")

        if stopper.step(acc, model):
            print("Early stopping triggered")
            break

    stopper.restore(model)
    return stopper.best_acc

# ---------------- Main Loop ----------------
BASE = "models/Data/Data/62_classes/UserIndependent"
margin_loss = MarginEmbeddingLoss(margin=1.0)
all_accs = []

for split in range(1, 11):
    print(f"\n========== SPLIT {split} ==========")
    train_ds = EMGDataset(f"{BASE}/Train/X_train_{split}.npy", f"{BASE}/Train/y_train_{split}.npy")
    test_ds = EMGDataset(f"{BASE}/Test/X_test_{split}.npy", f"{BASE}/Test/y_test_{split}.npy")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = FullModel(emb_dim=128).to(device)

    acc = train_and_eval(model, train_loader, test_loader, margin_loss, alpha=1.0, epochs=50, patience=10)
    print(f"Best Split Accuracy: {acc:.4f}")
    all_accs.append(acc)

    del model
    torch.cuda.empty_cache()

print("FINAL RESULT ")
print("Average Accuracy (Margin + CE):", np.mean(all_accs))
