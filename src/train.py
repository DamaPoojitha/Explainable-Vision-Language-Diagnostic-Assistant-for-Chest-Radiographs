"""
CPU training loop for the chest X-ray classifier.

Tips for keeping this feasible on CPU:
- Keep the dataset small (2,000-5,000 images).
- Use a small batch size (8-16) with num_workers=0 or 2.
- 5-10 epochs is usually enough since most of the network is frozen.
- Run this in the background (nohup / tmux) and work on Grad-CAM /
  report generation in parallel -- don't block your week on this step.
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from dataset import ChestXrayDataset, CONDITIONS, train_transform, eval_transform
from model import build_model


def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            all_preds.append(preds)
            all_targets.append(targets)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def train(
    csv_path="data/nih_sample/labels.csv",
    image_dir="data/nih_sample/images",
    epochs=8,
    batch_size=16,
    lr=1e-4,
    val_split=0.15,
    out_path="models/chest_classifier.pt",
):
    full_train_ds = ChestXrayDataset(csv_path, image_dir, transform=train_transform)
    full_eval_ds = ChestXrayDataset(csv_path, image_dir, transform=eval_transform)

    n_val = int(len(full_train_ds) * val_split)
    n_train = len(full_train_ds) - n_val
    generator = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(len(full_train_ds)), [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        torch.utils.data.Subset(full_train_ds, train_idx.indices),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(full_eval_ds, val_idx.indices),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model = build_model(num_classes=len(CONDITIONS))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / n_train
        metrics = evaluate(model, val_loader)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{epochs} | loss {avg_loss:.4f} | "
            f"val_acc {metrics['accuracy']:.3f} | val_f1 {metrics['f1']:.3f} | "
            f"{elapsed:.1f}s"
        )

    torch.save({"model_state": model.state_dict(), "conditions": CONDITIONS}, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/nih_sample/labels.csv")
    parser.add_argument("--images", default="data/nih_sample/images")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", default="models/chest_classifier.pt")
    args = parser.parse_args()

    train(
        csv_path=args.csv,
        image_dir=args.images,
        epochs=args.epochs,
        batch_size=args.batch_size,
        out_path=args.out,
    )