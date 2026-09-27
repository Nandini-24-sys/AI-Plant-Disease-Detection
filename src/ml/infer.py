import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import os
from torchvision import transforms

# -----------------------------------------------------
# FIXED MODEL PATH
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
print("MODEL PATH ->", MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file NOT found at: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    classes = checkpoint["classes"]
    num_classes = len(classes)

    # Build ResNet18
    model = models.resnet18(weights=None)

    # Replace FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load correct key: "state_dict"
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model, classes


# -----------------------------------------------------
# IMAGE PREPROCESSING
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


# -----------------------------------------------------
# GRAD-CAM
# -----------------------------------------------------
def generate_gradcam(model, input_tensor, save_path, raw_pil_image=None):
    gradients = None
    activations = None

    # Hooks
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    last_conv = model.layer4[1].conv2
    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_backward_hook(backward_hook)

    # Forward + Backward
    output = model(input_tensor)
    pred_idx = torch.argmax(output)
    model.zero_grad()
    output[0, pred_idx].backward()

    fh.remove()
    bh.remove()

    if gradients is None or activations is None:
        print("❌ GradCAM extraction failed")
        return None

    gradients = gradients.detach().cpu().numpy()[0]
    activations = activations.detach().cpu().numpy()[0]

    # Weight calculation
    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    # Convert original PIL image to OpenCV safely
    orig = np.array(raw_pil_image)
    orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

    # Resize CAM to original size
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

    # Overlay
    overlay = cv2.addWeighted(orig, 0.55, heatmap, 0.45, 0)

    cv2.imwrite(save_path, overlay)
    return save_path

# -----------------------------------------------------
# PREDICT WITH GRAD-CAM
# -----------------------------------------------------
def predict_with_gradcam(model, classes, image_path):
    pil_img = Image.open(image_path).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    prob = torch.softmax(outputs, dim=1)
    conf, predicted = torch.max(prob, 1)

    label = classes[predicted.item()]
    confidence = float(conf.item() * 100)

    gradcam_path = image_path + "_gradcam.jpg"
    generate_gradcam(model, tensor, gradcam_path, raw_pil_image=pil_img)

    return label, confidence, gradcam_path
