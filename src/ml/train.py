# src/ml/train.py

import os
import time
import copy
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------
# BUILD DATA LOADERS
# ---------------------------------------------------------
def build_dataloaders(data_dir, img_size=224, batch_size=32, num_workers=4):

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_set   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_set.classes


# ---------------------------------------------------------
# BUILD MODEL
# ---------------------------------------------------------
def build_model(num_classes, arch="resnet34", pretrained=True):

    if arch.startswith("resnet"):
        model = getattr(models, arch)(weights="IMAGENET1K_V1" if pretrained else None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)

    elif arch.startswith("efficientnet"):
        model = getattr(models, arch)(weights="IMAGENET1K_V1" if pretrained else None)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)

    else:
        raise ValueError("Unsupported architecture")

    return model


# ---------------------------------------------------------
# TRAIN ONE EPOCH
# ---------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    running_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------
@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()
    running_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Validation", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------
# MAIN TRAIN FUNCTION
# ---------------------------------------------------------
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Using Device: {device}")

    # Load data
    train_loader, val_loader, classes = build_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers
    )
    num_classes = len(classes)

    # Build model
    model = build_model(num_classes, arch=args.arch, pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, verbose=True)

    best_acc = 0.0

    # -----------------------------------------------------
    # RESUME TRAINING
    # -----------------------------------------------------
    if args.resume and os.path.exists(args.resume_path):
        print(f"\n🔄 Resuming from checkpoint → {args.resume_path}\n")
        ckpt = torch.load(args.resume_path, map_location=device)

        model.load_state_dict(ckpt["state_dict"])
        print("✔ Model weights restored")

        if "classes" in ckpt:
            print("✔ Classes loaded")

    # -----------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\n==============================")
        print(f"   Epoch {epoch}/{args.epochs}")
        print("==============================")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc

            torch.save({
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "classes": classes
            }, args.save_path)

            print(f"💾 Best model saved → {args.save_path} (Acc: {best_acc:.4f})")

    print("\n🎉 Training Finished!")


# ---------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        default=r"D:\plant disease app\data")

    parser.add_argument("--save_path", type=str,
                        default=r"D:\plant disease app\models\best_model.pth")

    parser.add_argument("--resume", action="store_true",
                        help="Resume training")

    parser.add_argument("--resume_path", type=str,
                        default=r"D:\plant disease app\models\best_model.pth")

    parser.add_argument("--arch", type=str, default="resnet34")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    main(args)
