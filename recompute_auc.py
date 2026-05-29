"""
Loads every saved checkpoint and computes accuracy, sensitivity, specificity,
and AUC-ROC on the test split.  Rewrites the Phase 3 pruning and distillation
reports with the new AUC-ROC column.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

from model import LightweightTBNet
from dataset import TBDataset, get_transforms
from metrics import evaluate

DATA_DIR = "data/"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

test_ds = TBDataset("test_split.csv", DATA_DIR, get_transforms(train=False))
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)


def load_teacher(path):
    m = LightweightTBNet(num_classes=2).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m


def load_student(path):
    m = mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device)


def load_fp16(path):
    m = LightweightTBNet(num_classes=2).to(torch.device("cpu"))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.half()


print("=== AUC-ROC for all checkpoints (test split) ===\n")
results = {}

for name, loader_fn, ckpt in [
    ("teacher (best_model)",     load_teacher, "checkpoints/best_model.pth"),
    ("pruned_25",                load_teacher, "checkpoints/pruned_25.pth"),
    ("pruned_50",                load_teacher, "checkpoints/pruned_50.pth"),
    ("pruned_75",                load_teacher, "checkpoints/pruned_75.pth"),
    ("student (fine-tuned)",     load_student, "checkpoints/student_best.pth"),
]:
    if not os.path.exists(ckpt):
        print(f"  SKIP {name}: {ckpt} not found")
        continue
    m = loader_fn(ckpt)
    dev = torch.device("cpu") if next(m.parameters()).dtype == torch.float16 else device
    acc, sens, spec, auc, lat = evaluate(m, test_loader, dev)
    results[name] = dict(acc=acc, sens=sens, spec=spec, auc=auc, lat=lat)
    print(f"  {name:<30}  Acc={acc*100:.2f}%  Sens={sens*100:.2f}%  Spec={spec*100:.2f}%  AUC={auc:.4f}")

# --- Patch pruning report ---
sparsity_rows = [
    ("0%",  "teacher (best_model)", 0.0),
    ("25%", "pruned_25",            0.25),
    ("50%", "pruned_50",            0.50),
    ("75%", "pruned_75",            0.75),
]

prune_report = "\n=== Phase 3 Pruning Report ===\n\n"
prune_report += f"{'Sparsity':<8}  {'Accuracy':>9}  {'Sensitivity':>11}  {'Specificity':>11}  {'AUC-ROC':>8}\n"
prune_report += "-" * 58 + "\n"

base = results.get("teacher (best_model)")
for label, key, _ in sparsity_rows:
    r = results.get(key)
    if r is None:
        continue
    prune_report += (f"  {label:<6}  {r['acc']*100:>8.2f}%  {r['sens']*100:>10.2f}%"
                     f"  {r['spec']*100:>10.2f}%  {r['auc']:>8.4f}\n")

if base:
    prune_report += "\nDelta from 0% baseline:\n"
    for label, key, _ in sparsity_rows[1:]:
        r = results.get(key)
        if r is None:
            continue
        prune_report += (f"  {label} pruned | Acc: {(r['acc']-base['acc'])*100:+.2f}%"
                         f" | Sens: {(r['sens']-base['sens'])*100:+.2f}%"
                         f" | Spec: {(r['spec']-base['spec'])*100:+.2f}%"
                         f" | AUC: {(r['auc']-base['auc']):+.4f}\n")

with open("phase3_pruning_report.txt", "w") as f:
    f.write(prune_report)
print("\n[saved] phase3_pruning_report.txt")

# --- Patch distillation report ---
teacher = results.get("teacher (best_model)")
student = results.get("student (fine-tuned)")

if teacher and student:
    sens_gap = abs(teacher["sens"] - student["sens"]) * 100
    within_2 = "YES" if sens_gap <= 2.0 else "NO"
    dist_report = f"""
=== Phase 3 Knowledge Distillation Report ===

Setup:
  Teacher:     LightweightTBNet (ResNet18 + AttentionCondenser)
  Student:     MobileNetV3-Small (sensitivity fine-tuned)
  Temperature: 4
  Alpha:       0.7 (distillation) / 0.3 (hard labels)

Test Set Results:
                    Teacher       Student     Delta
  Accuracy:        {teacher['acc']*100:>6.2f}%       {student['acc']*100:>6.2f}%     {(student['acc']-teacher['acc'])*100:+.2f}%
  Sensitivity:     {teacher['sens']*100:>6.2f}%       {student['sens']*100:>6.2f}%     {(student['sens']-teacher['sens'])*100:+.2f}%
  Specificity:     {teacher['spec']*100:>6.2f}%       {student['spec']*100:>6.2f}%     {(student['spec']-teacher['spec'])*100:+.2f}%
  AUC-ROC:         {teacher['auc']:>7.4f}       {student['auc']:>7.4f}     {(student['auc']-teacher['auc']):+.4f}
  Latency (ms):   {teacher['lat']:>7.2f}       {student['lat']:>7.2f}

Sensitivity Gap: {sens_gap:.2f}%
Student matches teacher sensitivity within 2%: {within_2}
"""
    with open("phase3_distillation_report.txt", "w") as f:
        f.write(dist_report)
    print("[saved] phase3_distillation_report.txt")
