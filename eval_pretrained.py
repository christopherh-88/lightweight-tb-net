"""
Sensitivity gap analysis: compare our from-scratch retrain against
the DarwinAI / TB-Net pretrained weights on the same test split.

Goal: determine whether any sensitivity delta is due to our retraining
or due to a different train/test split than the published paper.

Usage (two modes):
  1. Evaluate our model only (no download needed):
       python eval_pretrained.py

  2. Also evaluate DarwinAI pretrained weights (requires download — see
     the DARWINAI WEIGHTS section below):
       python eval_pretrained.py --darwinai path/to/weights.pth
"""

import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TBDataset, get_transforms
from model import LightweightTBNet
from metrics import evaluate as _evaluate

# ---------------------------------------------------------------------------
# Paper reference numbers (Wong et al. 2022, TB-Net)
# Official split sizes (from DarwinAI repo CSVs and README):
#   Train:  5,541 total (2,782 Normal + 2,759 TB)
#   Val:      693 total (  348 Normal +   345 TB)
#   Test:     693 total (  348 Normal +   345 TB)
# README: "The test dataset contains 348 normal samples and 345 tuberculosis samples."
# Paper test-set metrics: Sensitivity 100.0%, Specificity 99.71%, Acc 99.86%
#
# Dataset note: the Kaggle dataset used by DarwinAI originally had ~3,500 TB
# images. The current Kaggle version has only 700 TB images, so only 69 of the
# 345 TB test images are available on disk. See architecture_delta.txt.
# ---------------------------------------------------------------------------
PAPER = {
    "train_total": 5541, "train_tb": 2759, "train_normal": 2782,
    "val_total":    693,  "val_tb":   345,  "val_normal":    348,
    "test_total":   693,  "test_tb":  345,  "test_normal":   348,
    "sensitivity": 1.000,
    "specificity": 0.9971,
    "accuracy":    0.9986,
}

DATA_DIR = "data"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# DARWINAI WEIGHTS — how to obtain
# ---------------------------------------------------------------------------
# The official TB-Net weights are released at:
#   https://github.com/darwinai/TuberculosisNet
#   (see the "Models" section for download links)
#
# The original model is a TensorFlow/Keras HDF5 file.  To use it here:
#   Option A — convert to ONNX then load via onnxruntime:
#     pip install tf2onnx onnxruntime
#     python -m tf2onnx.convert --saved-model tb-net --output tb_net.onnx
#
#   Option B — if a PyTorch .pth is available (community conversion):
#     Pass it with --darwinai tb_net_pytorch.pth and implement
#     load_darwinai_model() below to match their architecture.
#
# If neither is available the script runs in "ours only" mode and still
# produces the class-balance comparison table and reproducibility report.
# ---------------------------------------------------------------------------


def load_darwinai_model(weights_path: str):
    """
    Stub — replace with actual DarwinAI architecture once weights are obtained.
    TB-Net uses projection-expansion-projection (PEP) blocks with self-attention
    on a lightweight backbone.  The exact layer sizes must match the checkpoint.
    Raises NotImplementedError until implemented.
    """
    raise NotImplementedError(
        "DarwinAI TB-Net PyTorch architecture not yet wired up.\n"
        "  1. Download weights from https://github.com/darwinai/TuberculosisNet\n"
        "  2. Implement the PEP-block architecture here to match the checkpoint\n"
        "     (or convert TF→ONNX and use eval_onnx() instead).\n"
        "  Running in 'ours only' mode."
    )


def evaluate(model, loader):
    from sklearn.metrics import confusion_matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    mat = confusion_matrix(all_labels, all_preds).astype(float)
    tn, fp, fn, tp = mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]
    acc  = (tp + tn) / mat.sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc_full, sens_full, spec_full, auc, _ = _evaluate(model, loader, device)
    return acc, sens, spec, auc, int(tp), int(fn), int(tn), int(fp)


def split_balance(csv_path, data_dir):
    """Return (total_loaded, n_normal, n_tb) for a split CSV."""
    labels = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            p = os.path.join(data_dir, row[0])
            if not os.path.exists(p):
                p = os.path.join(data_dir, row[0].replace("TB-", "Tuberculosis-"))
            if os.path.exists(p):
                labels.append(int(row[1]))
    c = Counter(labels)
    return len(labels), c[0], c[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--darwinai", metavar="PATH",
                        help="Path to DarwinAI TB-Net PyTorch weights (.pth)")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                        help="Our trained model checkpoint")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Class-balance comparison                                          #
    # ------------------------------------------------------------------ #
    our_train = split_balance("train_split.csv", DATA_DIR)
    our_val   = split_balance("val_split.csv",   DATA_DIR)
    our_test  = split_balance("test_split.csv",  DATA_DIR)

    print("\n=== Table 1: Dataset split comparison ===\n")
    header = f"{'Split':<8}  {'Total':>6}  {'Normal':>7}  {'TB':>6}  {'TB%':>6}"
    print(header)
    print("-" * len(header))

    def row(label, total, normal, tb):
        return f"{label:<8}  {total:>6}  {normal:>7}  {tb:>6}  {tb/total*100:>5.1f}%"

    print("--- Paper (Wong et al. 2022) ---")
    print(row("train",  PAPER["train_total"], PAPER["train_normal"], PAPER["train_tb"]))
    print(row("val",    PAPER["val_total"],   PAPER["val_normal"],   PAPER["val_tb"]))
    print(row("test",   PAPER["test_total"],  PAPER["test_normal"],  PAPER["test_tb"]))
    print("--- Ours (loaded from disk) ---")
    print(row("train",  *our_train))
    print(row("val",    *our_val))
    print(row("test",   *our_test))

    note_lines = []
    if our_test[0] != PAPER["test_total"]:
        delta = our_test[0] - PAPER["test_total"]
        note_lines.append(
            f"  * Our test set has {our_test[0]} loaded images vs {PAPER['test_total']} in paper "
            f"(delta {delta:+d}).  TB: ours {our_test[2]} vs paper {PAPER['test_tb']}."
        )
    if note_lines:
        print("\nNotes:")
        for n in note_lines:
            print(n)

    # ------------------------------------------------------------------ #
    # 2. Evaluate our model                                               #
    # ------------------------------------------------------------------ #
    print("\n=== Sensitivity evaluation ===\n")
    test_ds = TBDataset("test_split.csv", DATA_DIR, get_transforms(train=False))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    our_model = LightweightTBNet(num_classes=2).to(device)
    our_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    our_acc, our_sens, our_spec, our_auc, tp, fn, tn, fp = evaluate(our_model, test_loader)

    print(f"Model: our retrain ({args.checkpoint})")
    print(f"  Accuracy:    {our_acc*100:.2f}%   (paper: {PAPER['accuracy']*100:.2f}%)")
    print(f"  Sensitivity: {our_sens*100:.2f}%  (paper: {PAPER['sensitivity']*100:.2f}%)  "
          f"TP={tp} FN={fn}")
    print(f"  Specificity: {our_spec*100:.2f}%  (paper: {PAPER['specificity']*100:.2f}%)  "
          f"TN={tn} FP={fp}")
    print(f"  AUC-ROC:     {our_auc:.4f}")
    sens_delta = (our_sens - PAPER["sensitivity"]) * 100
    print(f"  Sensitivity delta vs paper: {sens_delta:+.2f}pp")

    results = {
        "ours_acc": our_acc,
        "ours_sens": our_sens,
        "ours_spec": our_spec,
        "ours_auc": our_auc,
        "darwinai_sens": None,
    }

    # ------------------------------------------------------------------ #
    # 3. Evaluate DarwinAI pretrained weights (optional)                  #
    # ------------------------------------------------------------------ #
    darwinai_sens = None
    if args.darwinai:
        print(f"\nModel: DarwinAI pretrained ({args.darwinai})")
        try:
            dai_model = load_darwinai_model(args.darwinai)
            dai_model = dai_model.to(device)
            dai_acc, dai_sens, dai_spec, tp2, fn2, tn2, fp2 = evaluate(dai_model, test_loader)
            print(f"  Accuracy:    {dai_acc*100:.2f}%")
            print(f"  Sensitivity: {dai_sens*100:.2f}%   TP={tp2} FN={fn2}")
            print(f"  Specificity: {dai_spec*100:.2f}%   TN={tn2} FP={fp2}")
            darwinai_sens = dai_sens
            results["darwinai_sens"] = dai_sens
        except NotImplementedError as e:
            print(f"  [SKIPPED] {e}")

    # ------------------------------------------------------------------ #
    # 4. Reproducibility interpretation & report                          #
    # ------------------------------------------------------------------ #
    lines = []
    lines.append("=== Sensitivity Gap Reproducibility Report (Wong et al. 2022) ===\n")

    lines.append("Split comparison (loaded images vs paper Table 1):")
    lines.append(f"  Our test:   {our_test[0]} images  ({our_test[2]} TB / {our_test[1]} Normal)")
    lines.append(f"  Paper test: {PAPER['test_total']} images  ({PAPER['test_tb']} TB / {PAPER['test_normal']} Normal)")
    lines.append(f"  TB%: ours {our_test[2]/our_test[0]*100:.1f}%  vs  paper {PAPER['test_tb']/PAPER['test_total']*100:.1f}%\n")

    lines.append("Our retrain on this split:")
    lines.append(f"  Sensitivity: {our_sens*100:.2f}%  (paper: {PAPER['sensitivity']*100:.2f}%,  delta: {sens_delta:+.2f}pp)")
    lines.append(f"  Specificity: {our_spec*100:.2f}%  (paper: {PAPER['specificity']*100:.2f}%)")
    lines.append(f"  Accuracy:    {our_acc*100:.2f}%   (paper: {PAPER['accuracy']*100:.2f}%)")
    lines.append(f"  AUC-ROC:     {our_auc:.4f}\n")

    if darwinai_sens is not None:
        dai_delta = (darwinai_sens - PAPER["sensitivity"]) * 100
        our_vs_dai = (our_sens - darwinai_sens) * 100
        lines.append("DarwinAI pretrained weights on our split:")
        lines.append(f"  Sensitivity: {darwinai_sens*100:.2f}%  (delta vs paper: {dai_delta:+.2f}pp)")
        lines.append(f"  Sensitivity delta (ours vs DarwinAI): {our_vs_dai:+.2f}pp\n")

        if abs(dai_delta) < 0.5:
            lines.append("Interpretation: DarwinAI pretrained weights reproduce paper sensitivity")
            lines.append("on our split. Any remaining delta between our retrain and the paper is")
            lines.append("attributable to our from-scratch training (seed / aug / LR schedule).")
        else:
            lines.append("Interpretation: DarwinAI pretrained weights ALSO deviate from paper")
            lines.append("sensitivity on our split. The gap is primarily due to split mismatch")
            lines.append("(our images are a subset of a larger dataset; the paper used a different")
            lines.append("train/test partition — see Table 1 delta above).")
    else:
        lines.append("DarwinAI pretrained weights: not evaluated (download required).")
        lines.append("To complete this analysis:")
        lines.append("  1. Download weights from https://github.com/darwinai/TuberculosisNet")
        lines.append("  2. Run: python eval_pretrained.py --darwinai <path>")

        # Provide interpretation based on our result alone
        lines.append("")
        if abs(sens_delta) < 0.5:
            lines.append("Preliminary finding (our retrain only):")
            lines.append("  Our from-scratch retrain matches paper sensitivity on this split.")
            lines.append("  This suggests the split composition is compatible with the paper's")
            lines.append("  test set (similar TB count and image distribution).")
        else:
            lines.append("Preliminary finding (our retrain only):")
            lines.append(f"  Our retrain shows a {sens_delta:+.2f}pp sensitivity delta vs the paper.")
            lines.append("  Possible causes:")
            lines.append("  a) Retraining: seed, augmentation policy, or LR schedule difference")
            lines.append("  b) Split mismatch: our test set has different TB count than paper")
            lines.append(f"     (ours: {our_test[2]} TB images vs paper: {PAPER['test_tb']})")
            lines.append("  Run DarwinAI weights to disambiguate.")

    report = "\n".join(lines)
    print("\n" + report)

    with open("sensitivity_gap_report.txt", "w") as f:
        f.write(report + "\n")
    print("\n[saved] sensitivity_gap_report.txt")


if __name__ == "__main__":
    main()
