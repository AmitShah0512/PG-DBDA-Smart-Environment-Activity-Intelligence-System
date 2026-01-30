import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from utils.config import DATASET_PATH, MODEL_PATH
from model.net import CNNBiLSTM

def train_model():
    print("Loading dataset...")
    data = np.load(DATASET_PATH, allow_pickle=True)

    X = data["X"]
    y = data["y"]

    if "class_names" in data:
        class_names = list(data["class_names"])
    elif "label_map" in data:
        inv = {v: k for k, v in data["label_map"].item().items()}
        class_names = [inv[i] for i in range(len(inv))]
    else:
        raise RuntimeError("No class info found")

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = CNNBiLSTM(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 60
    print("Starting training...")

    for ep in range(epochs):
        correct = 0
        total = 0
        loss_sum = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        acc = 100 * correct / total
        print(f"Epoch {ep+1}/{epochs} | Loss: {loss_sum/len(loader):.4f} | Accuracy: {acc:.2f}%")

    torch.save({
        "model_state": model.state_dict(),
        "class_names": class_names
    }, MODEL_PATH)

    print("Saved:", MODEL_PATH)
