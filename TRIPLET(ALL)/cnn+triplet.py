import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from pytorch_metric_learning import losses, miners, distances

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
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )

# ---------------- CNN Encoder (LEARNED VERSION) ----------------
class CNNEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.features = nn.Sequential(
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
        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (B, C, T)
        x = self.features(x).squeeze(-1)
        z = self.fc(x)
        return z                        # ❗ no normalize here

# ---------------- Full Model ----------------
class FullModel(nn.Module):
    def __init__(self, emb_dim=256, num_classes=62):
        super().__init__()
        self.encoder = CNNEncoder(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return z, logits

# ---------------- Training ----------------
def train_and_eval(model, train_loader, test_loader,
                   metric_loss, miner,
                   epochs=100, patience=10):

    ce_loss = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=5e-4)

    best_acc = -1
    patience_ctr = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            embeddings, logits = model(x)

            indices_tuple = miner(embeddings, y)
            loss = (
                metric_loss(embeddings, y, indices_tuple)
                + ce_loss(logits, y)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------- Evaluation --------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                _, logits = model(x)
                preds = logits.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1:03d} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping")
                break

    model.load_state_dict(best_state)
    return best_acc

# ---------------- Metric Losses (REFERENCE-STYLE) ----------------
encoder_losses = {
    "Triplet_L2": losses.TripletMarginLoss(
        margin=0.2,
        distance=distances.LpDistance(normalize_embeddings=True, p=2, power=2),
        smooth_loss=True
    ),
    "Triplet_Cosine": losses.TripletMarginLoss(
        margin=0.1,
        distance=distances.CosineSimilarity(),
        smooth_loss=True
    ),
    "NTXent": losses.NTXentLoss(temperature=0.2)
}

# ---------------- Miner (KEY FIX) ----------------
miner = miners.BatchEasyHardMiner(
    pos_strategy="semihard",
    neg_strategy="hard"
)

# ---------------- Experiment Loop ----------------
BASE = "models/Data/Data/62_classes/UserIndependent"
final_results = {}

for loss_name, metric_loss in encoder_losses.items():
    print(f"\n==============================")
    print(f" Encoder Loss: {loss_name}")
    print(f"==============================")

    accs = []

    for split in range(1, 11):
        print(f"\n----- SPLIT {split} -----")

        train_ds = EMGDataset(
            f"{BASE}/Train/X_train_{split}.npy",
            f"{BASE}/Train/y_train_{split}.npy"
        )
        test_ds = EMGDataset(
            f"{BASE}/Test/X_test_{split}.npy",
            f"{BASE}/Test/y_test_{split}.npy"
        )

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        model = FullModel().to(device)

        acc = train_and_eval(
            model,
            train_loader,
            test_loader,
            metric_loss,
            miner,
            epochs=100,
            patience=10
        )

        print(f"Split Accuracy: {acc:.4f}")
        accs.append(acc)

        del model
        torch.cuda.empty_cache()

    final_results[loss_name] = np.mean(accs)
    print(f"\n>>> Average Accuracy ({loss_name}): {final_results[loss_name]:.4f}")

print("\n================ FINAL RESULTS ================")
for k, v in final_results.items():
    print(f"{k:20s} : {v:.4f}")