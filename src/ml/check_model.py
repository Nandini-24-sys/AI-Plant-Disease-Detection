import torch
import os

MODEL_PATH = r"D:\plant disease app\src\models\best_model.pth"

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

print("\n🔍 KEYS FOUND IN MODEL FILE:")
print(checkpoint.keys())
