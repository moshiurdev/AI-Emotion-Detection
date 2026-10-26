"""Evaluate a trained checkpoint on the FER-2013 PrivateTest split.

Writes ``metrics.json``, ``classification_report.txt``, ``confusion_matrix.png``
and (if the run has a ``history.json``) ``training_curves.png`` to ``--out``.

Example:
    python evaluate.py --checkpoint runs/resnet18/best.pt --out results/resnet18
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from fer import EMOTIONS
from fer.data import FER2013
from fer.models import count_parameters
from fer.utils import load_checkpoint, pick_device


@torch.no_grad()
def predict(model, loader, device, tta: bool) -> tuple[np.ndarray, np.ndarray]:
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device)
        p = model(x).softmax(1)
        if tta:  # average with the horizontally mirrored face
            p = (p + model(x.flip(-1)).softmax(1)) / 2
        probs.append(p.cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def summarise(probs: np.ndarray, targets: np.ndarray) -> dict:
    preds = probs.argmax(1)
    report = classification_report(targets, preds, target_names=EMOTIONS, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro"),
        "weighted_f1": f1_score(targets, preds, average="weighted"),
        "per_class": {
            name: {k: report[name][k] for k in ("precision", "recall", "f1-score", "support")}
            for name in EMOTIONS
        },
    }


@torch.no_grad()
def cpu_latency_ms(model, runs: int = 50) -> float:
    """Median single-face forward pass on CPU, i.e. the cost per face in the web app."""
    model = model.to("cpu").eval()
    x = torch.rand(1, 1, 48, 48)
    for _ in range(5):
        model(x)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    cm_norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised (recall)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(epochs, [h["train_loss"] for h in history], label="train (weighted, smoothed CE)")
    a1.plot(epochs, [h["val_loss"] for h in history], label="val (plain CE)")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(epochs, [h["train_acc"] for h in history], label="train acc (augmented)")
    a2.plot(epochs, [h["val_acc"] for h in history], label="val acc")
    a2.plot(epochs, [h["val_macro_f1"] for h in history], label="val macro-F1", linestyle="--")
    a2.set_xlabel("epoch"); a2.set_ylabel("score"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a FER-2013 checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    loader = DataLoader(FER2013(args.split), batch_size=256, shuffle=False)
    args.out.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": str(args.checkpoint),
        "split": {"val": "PublicTest", "test": "PrivateTest"}[args.split],
        "arch": ckpt["config"]["arch"],
        "input_size": ckpt["config"]["input_size"],
        "params": count_parameters(model),
        "best_epoch": ckpt.get("epoch"),
    }
    for tta in (False, True):
        probs, targets = predict(model, loader, device, tta)
        key = "tta_hflip" if tta else "single"
        results[key] = summarise(probs, targets)
        print(f"[{key}] acc {results[key]['accuracy']:.4f}  macro-F1 {results[key]['macro_f1']:.4f}")
        if tta:
            preds = probs.argmax(1)
            cm = confusion_matrix(targets, preds, labels=range(len(EMOTIONS)))
            results["confusion_matrix"] = cm.tolist()
            plot_confusion(cm, args.out / "confusion_matrix.png",
                           f"{results['arch']} on FER-2013 {results['split']} "
                           f"(acc {results[key]['accuracy']:.3f}, hflip TTA)")
            (args.out / "classification_report.txt").write_text(
                classification_report(targets, preds, target_names=EMOTIONS, digits=4, zero_division=0))

    results["cpu_latency_ms_per_face"] = cpu_latency_ms(model)
    print(f"CPU latency: {results['cpu_latency_ms_per_face']:.1f} ms / face")

    history_path = args.checkpoint.parent / "history.json"
    if history_path.exists():
        plot_history(json.loads(history_path.read_text()), args.out / "training_curves.png")

    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
