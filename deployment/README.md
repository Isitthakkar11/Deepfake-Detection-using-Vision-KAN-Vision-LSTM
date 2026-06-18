# Optional: compile the model for edge runtimes (IREE / MLIR)

These scripts are **not required** to run the demo. They demonstrate exporting the
trained model to a portable, Python-free binary using
[IREE](https://iree.dev/) (an MLIR-based ML compiler).

| Script                     | What it does                                                                 |
| -------------------------- | --------------------------------------------------------------------------- |
| `compile_deepfake_iree.py` | Rebuilds the real Vision-KAN + Vision-LSTM model, exports to ONNX → MLIR, compiles to a `.vmfb`, and benchmarks PyTorch vs IREE. |
| `compile_vit_iree.py`      | A clean end-to-end sanity pipeline on a standard DeiT-tiny ViT — proves the full PyTorch → ONNX → MLIR → IREE → `.vmfb` path compiles and runs. |

### Setup

```bash
pip install iree-compiler iree-runtime --break-system-packages
```

Run either script from a folder that also contains `as_model_best.pt`:

```bash
python compile_deepfake_iree.py
```

### Note

The generated artifacts (`*.onnx`, `*.mlir`, `*.vmfb`, `*_onnx.data`) are large
intermediate files and are intentionally git-ignored. Regenerate them with the
scripts above rather than committing them.
