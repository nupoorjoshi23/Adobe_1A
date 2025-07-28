# src/optimize.py (UPDATED for ELECTRA model paths)
import torch
from src.model import MiniLayoutLM

# --- Configuration ---
TRAINED_MODEL_DIR = 'models/electra-layoutlm-full'
QUANTIZED_MODEL_PATH = 'models/electra-layoutlm-full-quantized.pth'

print(f"Loading model from {TRAINED_MODEL_DIR}...")
model = MiniLayoutLM.from_pretrained(TRAINED_MODEL_DIR)
model.eval()

print("Applying dynamic quantization for CPU...")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

print(f"Saving quantized model to {QUANTIZED_MODEL_PATH}...")
torch.save(quantized_model.state_dict(), QUANTIZED_MODEL_PATH)

print("✅ Optimization complete.")