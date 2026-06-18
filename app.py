"""
Interactive deepfake detector — upload a face image, get REAL / FAKE.

Run locally:
    pip install -r requirements.txt
    python app.py        # then open the printed local URL

The decision logic is unchanged from the original notebook:
    sigmoid output >= 0.5  ->  REAL
    sigmoid output <  0.5  ->  FAKE
"""

import os
import urllib.request

import gradio as gr
from PIL import Image

from src.model import load_model, predict

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "as_model_best.pt")
WEIGHTS_URL = os.environ.get("WEIGHTS_URL", "")  # optional: auto-download on a Space


def _ensure_weights():
    """If weights are missing and a WEIGHTS_URL is provided, fetch them once."""
    if not os.path.exists(WEIGHTS_PATH) and WEIGHTS_URL:
        print(f"Downloading weights from {WEIGHTS_URL} ...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        print("Done.")


print("Loading Vision-KAN + Vision-LSTM model (first load downloads backbones)...")
_ensure_weights()
MODEL = load_model(WEIGHTS_PATH)
print("Model ready.")


def classify(image: Image.Image):
    if image is None:
        return {"Upload an image": 1.0}, ""
    label, prob = predict(MODEL, image)
    # prob is P(REAL). Show both classes so the UI renders confidence bars.
    scores = {"REAL": float(prob), "FAKE": float(1.0 - prob)}
    confidence = prob if label == "REAL" else 1.0 - prob
    verdict = f"### Prediction: **{label}**  \nConfidence: **{confidence * 100:.1f}%**"
    return scores, verdict


DESCRIPTION = """
# 🕵️ Deepfake Detector — Vision-KAN + Vision-LSTM

Upload a **face image** and the hybrid model predicts whether it is **real** or a **deepfake**.

This is a frame-level detector that fuses two backbones — a **Vision-KAN**
transformer (spatial structure) and a **Vision-LSTM** (token-sequence modeling) —
via late fusion. Trained on a DFDC-derived dataset of extracted face frames.

> ⚠️ Research/educational demo. Works best on tightly-cropped frontal faces
> (the model was trained on DFDC face crops). It is not a production forensic tool.
"""

ARTICLE = """
---
**How it works:** image → resize 224×224 → ImageNet normalize → Vision-KAN logit
& Vision-LSTM logit → concatenate → MLP → sigmoid → threshold at 0.5.

Built by **Isit Thakkar** & **Krushna Bhujbal** · UTA CSE-6367 Computer Vision.
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), title="Deepfake Detector") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Upload a face image", height=320)
            btn = gr.Button("Detect", variant="primary")
            gr.Examples(
                examples=[
                    ["examples/real_sample.jpg"],
                    ["examples/fake_sample.jpg"],
                ],
                inputs=inp,
                label="Try an example",
            )
        with gr.Column(scale=1):
            verdict = gr.Markdown()
            out = gr.Label(num_top_classes=2, label="Scores")
    gr.Markdown(ARTICLE)

    btn.click(classify, inputs=inp, outputs=[out, verdict])
    inp.change(classify, inputs=inp, outputs=[out, verdict])


if __name__ == "__main__":
    demo.launch()
