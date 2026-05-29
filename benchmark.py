import time
import platform
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import LightweightTBNet
from dataset import TBDataset, get_transforms
from metrics import evaluate

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = LightweightTBNet(num_classes=2).to(device)
model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
model.eval()

test_dataset = TBDataset('test_split.csv', 'data/', get_transforms(train=False))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

total_params = sum(p.numel() for p in model.parameters())
model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

# Warm-up + latency timing (separate pass so evaluate() gives clean per-image ms)
latencies = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        t0 = time.perf_counter()
        model(imgs)
        latencies.append((time.perf_counter() - t0) * 1000)

acc, sensitivity, specificity, auc, _ = evaluate(model, test_loader, device)

report = f"""
=== Phase 2 Benchmark Report ===

System Info:
  OS:          {platform.system()} {platform.release()}
  Processor:   {platform.processor() or 'Apple Silicon'}
  Device:      {device}
  CUDA:        {'Available: ' + torch.version.cuda if torch.cuda.is_available() else 'Not available (CPU/MPS)'}
  PyTorch:     {torch.__version__}

Model Info:
  Parameters:  {total_params:,} ({total_params/1e6:.2f}M)
  Model size:  {model_size_mb:.2f} MB (FP32)

Inference Latency (per image):
  Mean:        {np.mean(latencies):.2f} ms
  Median:      {np.median(latencies):.2f} ms
  Min:         {np.min(latencies):.2f} ms
  Max:         {np.max(latencies):.2f} ms

Test Set Results:
  Accuracy:    {acc*100:.2f}%  (paper: 99.86%)
  Sensitivity: {sensitivity*100:.2f}%  (paper: 100.00%)
  Specificity: {specificity*100:.2f}%  (paper: 99.71%)
  AUC-ROC:     {auc:.4f}

Delta from Paper:
  Accuracy:    {(acc - 0.9986)*100:+.2f}%
  Sensitivity: {(sensitivity - 1.0)*100:+.2f}%
  Specificity: {(specificity - 0.9971)*100:+.2f}%
"""

print(report)
with open('phase2_report.txt', 'w') as f:
    f.write(report)
print('Saved to phase2_report.txt')
