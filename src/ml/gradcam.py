import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, input_tensor, class_idx=None):
        # input_tensor: 1xCxHxW
        outputs = self.model(input_tensor)  # forward
        if class_idx is None:
            class_idx = outputs.argmax().item()
        loss = outputs[0, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()
        weights = np.mean(grads, axis=(1,2))  # global avg pool
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))
        cam = (cam - cam.min()) / (cam.max()-cam.min() + 1e-8)
        return cam, class_idx
