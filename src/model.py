"""
Vision-KAN + Vision-LSTM hybrid deepfake detector.

This module is a faithful, refactored copy of the inference path in
`notebooks/test_mac.ipynb`. The architecture, weights loading, preprocessing,
and the real/fake decision rule are IDENTICAL to the original notebook — only
reorganized into importable functions so the demo app and the notebook share
one source of truth.

Decision rule (unchanged from the original project):
    prob = sigmoid output of the model
    prob >= 0.5  -> REAL
    prob <  0.5  -> FAKE
"""

import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

DEVICE = torch.device("cpu")

# ImageNet normalization — exactly as used during training/inference.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# ──────────────────────────────────────────────────────────────────────────
# Model definition (unchanged from test_mac.ipynb)
# ──────────────────────────────────────────────────────────────────────────
class CombinedModel(nn.Module):
    """Late-fusion of a Vision-KAN backbone and a Vision-LSTM backbone.

    Each backbone emits a scalar logit. The two logits are concatenated and
    passed through a small MLP, then a sigmoid produces a probability in [0, 1].
    """

    def __init__(self, KAN, vision_lstm):
        super(CombinedModel, self).__init__()
        self.KAN = KAN
        self.vision_lstm = vision_lstm
        self.fc1 = nn.Linear(2, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        KAN_output = self.KAN(x)
        lstm_output = self.vision_lstm(x)
        combined = torch.cat((KAN_output, lstm_output), dim=1)
        x = self.fc1(combined)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# ──────────────────────────────────────────────────────────────────────────
# Build + load
# ──────────────────────────────────────────────────────────────────────────
def build_model():
    """Reconstruct the exact Vision-KAN + Vision-LSTM hybrid (random weights)."""
    from VisionKAN import create_model

    KAN = create_model(
        model_name="deit_tiny_patch16_224_KAN",
        pretrained=False,
        hdim_kan=192,
        num_classes=1,
        drop_rate=0.0,
        drop_path_rate=0.05,
        img_size=224,
        batch_size=32,
    )

    vision_lstm = torch.hub.load(
        "nx-ai/vision-lstm",
        "VisionLSTM",
        dim=192,
        depth=24,
        patch_size=16,
        input_shape=(3, 224, 224),
        output_shape=(1,),
        drop_path_rate=0.05,
        stride=None,
    )

    return CombinedModel(KAN, vision_lstm).to(DEVICE)


def load_model(weights_path="as_model_best.pt"):
    """Build the model and load trained weights, returning it in eval mode."""
    model = build_model()

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Trained weights not found at '{weights_path}'. "
            f"Download as_model_best.pt (see README) and place it here."
        )

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────
# Preprocessing (unchanged from test_mac.ipynb -> preprocess_pil)
# ──────────────────────────────────────────────────────────────────────────
def preprocess_pil(img: Image.Image) -> torch.Tensor:
    """Resize to 224x224, scale to [0,1], ImageNet-normalize, return CHW tensor."""
    img = img.convert("RGB").resize((224, 224))
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    img_np = (img_np - _MEAN) / _STD
    img_np = img_np.transpose(2, 0, 1)
    return torch.tensor(img_np, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, img: Image.Image):
    """Run the model on a PIL image.

    Returns (label, probability) where probability is the raw sigmoid output
    and label is decided with the original threshold: prob >= 0.5 -> REAL.
    """
    x = preprocess_pil(img).unsqueeze(0).to(DEVICE)
    prob = model(x).item()
    label = "REAL" if prob >= 0.5 else "FAKE"
    return label, prob
