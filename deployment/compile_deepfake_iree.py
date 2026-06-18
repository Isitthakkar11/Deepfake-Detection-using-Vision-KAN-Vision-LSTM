"""
compile_deepfake_iree.py
Run this on YOUR MacBook, in the same environment/folder as test_mac.ipynb
(where VisionKAN, vision-lstm, and as_model_best.pt already work)

Step 0 — one-time install (run in terminal first):
    pip install iree-compiler iree-runtime --break-system-packages

Then run this file:
    python3 compile_deepfake_iree.py
"""

import torch
import torch.nn as nn
import os
import time

device = torch.device('cpu')

# ── Step 1: Rebuild your exact model (same as test_mac.ipynb) ──────────────
from VisionKAN import create_model

KAN = create_model(
    model_name='deit_tiny_patch16_224_KAN',
    pretrained=False,
    hdim_kan=192,
    num_classes=1,
    drop_rate=0.0,
    drop_path_rate=0.05,
    img_size=224,
    batch_size=32
)

vision_lstm = torch.hub.load(
    "nx-ai/vision-lstm", "VisionLSTM",
    dim=192,
    depth=24,
    patch_size=16,
    input_shape=(3, 224, 224),
    output_shape=(1,),
    drop_path_rate=0.05,
    stride=None
)

class CombinedModel(nn.Module):
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

model = CombinedModel(KAN, vision_lstm).to(device)

weights_path = "as_model_best.pt"
if os.path.exists(weights_path):
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ Loaded trained weights from {weights_path}")
else:
    raise FileNotFoundError(
        f"{weights_path} not found. Put this script in the same folder "
        f"as as_model_best.pt before running."
    )

model.eval()

# ── FIX: globally patch F.layer_norm so NO onnx.LayerNormalization is emitted ─
# The module-swap approach misses LayerNorms that are called functionally
# (F.layer_norm) inside timm/VisionKAN — there's no nn.LayerNorm module to
# swap. Patching the function itself catches every single case: module-based,
# functional, bias-less, all of them. We decompose into primitive ops that
# ONNX + IREE handle natively, so onnx.LayerNormalization never appears.
import torch.nn.functional as F

_orig_layer_norm = F.layer_norm

def _manual_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    dims = tuple(range(-len(normalized_shape), 0))
    mean = input.mean(dim=dims, keepdim=True)
    var = input.var(dim=dims, keepdim=True, unbiased=False)
    out = (input - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out

F.layer_norm = _manual_layer_norm
print("✓ Patched F.layer_norm globally (catches functional + module LayerNorms)")
model.eval()

# ── Step 2: Baseline — measure PyTorch inference time first ────────────────
dummy_input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    # warmup
    for _ in range(3):
        _ = model(dummy_input)
    # timed runs
    start = time.time()
    for _ in range(10):
        pytorch_output = model(dummy_input)
    pytorch_time = (time.time() - start) / 10

print(f"\n--- PyTorch baseline ---")
print(f"Output: {pytorch_output.item():.4f}")
print(f"Avg inference time: {pytorch_time*1000:.2f} ms")

# Model size on disk
pt_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
print(f"Weights file size: {pt_size_mb:.2f} MB")

# ── Step 3: Export to TorchScript (required before IREE import) ────────────
print(f"\n--- Exporting to TorchScript ---")
scripted_model = torch.jit.trace(model, dummy_input)
scripted_model.save("deepfake_model.pt")
print("✓ Saved deepfake_model.pt (TorchScript)")

# ── Step 4a: Export to ONNX (legacy exporter so the patch is respected) ─────
print(f"\n--- Exporting to ONNX ---")
try:
    onnx_path = "deepfake_model.onnx"

    # Export from the TRACED model: tracing already baked in our patched
    # F.layer_norm as primitive ops, so no LayerNormalization op can appear.
    # dynamo=False forces the legacy exporter which respects the trace.
    torch.onnx.export(
        scripted_model,             # traced model with patched layernorm baked in
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,               # legacy TorchScript exporter
    )

    onnx_size_kb = os.path.getsize(onnx_path) / 1024
    print(f"✓ Saved {onnx_path}")
    print(f"  ONNX file size: {onnx_size_kb:.1f} KB")

except Exception as e:
    print(f"\n✗ ONNX export failed with error:")
    print(f"  {type(e).__name__}: {e}")
    print(f"\nCopy this EXACT error back to Claude. Do not fabricate a result.")
    raise

# ── Step 4b: Convert ONNX -> MLIR via IREE's ONNX importer ──────────────────
print(f"\n--- Importing ONNX into MLIR (iree.compiler.tools.import_onnx) ---")
try:
    import subprocess, sys

    mlir_text_path = "deepfake_model.mlir"
    result = subprocess.run(
        [sys.executable, "-m", "iree.compiler.tools.import_onnx",
         onnx_path, "-o", mlir_text_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"\n✗ ONNX import failed:")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        raise RuntimeError("ONNX->MLIR import failed — see output above")

    mlir_size_kb = os.path.getsize(mlir_text_path) / 1024
    print(f"✓ Saved {mlir_text_path}")
    print(f"  MLIR text file size: {mlir_size_kb:.1f} KB")

except Exception as e:
    print(f"\n✗ ONNX->MLIR import failed with error:")
    print(f"  {type(e).__name__}: {e}")
    print(f"\nCopy this EXACT error back to Claude. Do not fabricate a result.")
    raise

# ── Step 4c: Compile the MLIR through IREE ──────────────────────────────────
# With LayerNorm decomposed at the source, there's no illegal onnx op left,
# so we can compile the imported MLIR directly.
print(f"\n--- Compiling through IREE ---")
import iree.compiler as ireec
try:
    compiled_flatbuffer = ireec.compile_file(
        mlir_text_path,
        input_type="onnx",
        target_backends=["llvm-cpu"],
    )

    vmfb_path = "deepfake_iree.vmfb"
    with open(vmfb_path, "wb") as f:
        f.write(compiled_flatbuffer)

    vmfb_size_kb = os.path.getsize(vmfb_path) / 1024
    print(f"✓ Compiled successfully: {vmfb_path}")
    print(f"  .vmfb size: {vmfb_size_kb:.1f} KB")

except Exception as e:
    print(f"\n✗ IREE compilation failed with error:")
    print(f"  {type(e).__name__}: {e}")
    print(f"\nIf this names a NEW illegal op (not LayerNormalization), paste it")
    print(f"to Claude. Do not fabricate a result.")
    raise

# ── Step 5: Run inference on the compiled binary ────────────────────────────
print(f"\n--- Running inference via IREE Runtime ---")
import iree.runtime as ireert
import numpy as np

input_np = dummy_input.numpy().astype(np.float32)

try:
    config = ireert.Config("local-task")
    with open(vmfb_path, "rb") as f:
        vmfb_bytes = f.read()
    vm_module = ireert.load_vm_flatbuffer(vmfb_bytes, driver="local-task")

    # ONNX-imported models usually expose 'main_graph'; fall back to others
    fn_name = None
    for candidate in ["main_graph", "main", "forward"]:
        if candidate in vm_module.vm_module.function_names:
            fn_name = candidate
            break
    if fn_name is None:
        # last resort: list what's available
        print(f"  Available functions: {vm_module.vm_module.function_names}")
        fn_name = [n for n in vm_module.vm_module.function_names
                   if not n.startswith("__")][0]

    print(f"  Calling function: {fn_name}")
    entry = getattr(vm_module, fn_name)

    # warmup + timed
    for _ in range(3):
        _ = entry(input_np)
    start = time.time()
    for _ in range(10):
        iree_output = entry(input_np)
    iree_time = (time.time() - start) / 10

    print(f"IREE output: {np.asarray(iree_output).flatten()[:5]}")
    print(f"Avg IREE inference time: {iree_time*1000:.2f} ms")

    # ── Step 6: Final comparison summary ────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"SUMMARY — PyTorch vs IREE")
    print(f"{'='*50}")
    print(f"PyTorch output         : {pytorch_output.item():.4f}")
    print(f"PyTorch inference time : {pytorch_time*1000:.2f} ms")
    print(f"IREE inference time    : {iree_time*1000:.2f} ms")
    print(f"Speedup                : {pytorch_time/iree_time:.2f}x")
    print(f"PyTorch weights size   : {pt_size_mb:.2f} MB")
    print(f".vmfb compiled size    : {vmfb_size_kb:.1f} KB")
    print(f"{'='*50}")
    print(f"\nSave this entire output. This is your real, verifiable result.")

except Exception as e:
    print(f"\n⚠ Compile SUCCEEDED but inference hit an API issue:")
    print(f"  {type(e).__name__}: {e}")
    print(f"\nThe .vmfb compiled fine — that's the hard part and it's done.")
    print(f"Paste this error to Claude to fix the inference call.")
    raise