"""Shared evaluation utilities used by all scripts."""
import time
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score


def evaluate(model, loader, device):
    """Return (acc, sensitivity, specificity, auc, mean_latency_ms)."""
    model.eval()
    all_preds, all_labels, all_probs, latencies = [], [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if next(model.parameters()).dtype == torch.float16:
                imgs = imgs.half()
            t0 = time.perf_counter()
            logits = model(imgs)
            latencies.append((time.perf_counter() - t0) * 1000 / imgs.size(0))
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    mat = confusion_matrix(all_labels, all_preds).astype(float)
    tn, fp, fn, tp = mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]
    acc  = (tp + tn) / mat.sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    auc  = roc_auc_score(all_labels, all_probs)
    return acc, sens, spec, auc, float(np.mean(latencies))
