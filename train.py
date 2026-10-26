"""Train an expression classifier on FER-2013.

Model selection uses the PublicTest split (``val``); the PrivateTest split
(``test``) is only touched by ``evaluate.py``.

Examples:
    python train.py --arch resnet18 --input-size 96 --epochs 30 --name resnet18
    python train.py --arch small_cnn --epochs 30 --name small_cnn
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from fer.data import FER2013, class_weights, eval_transform, train_transform
from fer.models import build_model, count_parameters
from fer.utils import pick_device, save_checkpoint, seed_everything

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a FER-2013 classifier")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small", "small_cnn"])
    p.add_argument("--input-size", type=int, default=96, help="backbone input resolution (pretrained archs)")
    p.add_argument("--no-pretrained", action="store_true", help="train a backbone from random init")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01, help="peak SGD learning rate")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--class-weighting", default="sqrt", choices=["none", "sqrt", "inverse"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0, help="use only N train/val samples (smoke tests)")
    p.add_argument("--name", default=None, help="run directory name under runs/")
    return p.parse_args()


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    """Linear warmup followed by cosine decay to zero."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(F.cross_entropy(logits, y, reduction="sum").item())
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": sum(losses) / len(targets),
        "acc": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    run_dir = ROOT / "runs" / (args.name or f"{args.arch}_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FER2013("train", transform=train_transform())
    val_ds = FER2013("val", transform=eval_transform())
    weights = class_weights(train_ds.labels, args.class_weighting).to(device)
    if args.limit:
        train_ds = Subset(train_ds, range(min(args.limit, len(train_ds))))
        val_ds = Subset(val_ds, range(min(args.limit, len(val_ds))))

    loader_kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, **loader_kw)

    model = build_model(args.arch, input_size=args.input_size, pretrained=not args.no_pretrained,
                        dropout=args.dropout).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True,
                                weight_decay=args.weight_decay)

    config = {k: v for k, v in vars(args).items() if k not in {"workers", "device", "name"}}
    config["params"] = count_parameters(model)
    config["class_weights"] = [round(w, 4) for w in weights.tolist()]
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"run dir: {run_dir}\ndevice: {device}  params: {config['params']:,}")
    print(f"class weights ({args.class_weighting}): {config['class_weights']}")

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    history, best_acc, step = [], -1.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, run_correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, warmup_steps, args.lr)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        val = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": run_loss / seen,
            "train_acc": run_correct / seen,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "time_s": time.time() - t0,
        }
        history.append(row)
        is_best = val["acc"] > best_acc
        if is_best:
            best_acc = val["acc"]
            save_checkpoint(run_dir / "best.pt", model, config, {"epoch": epoch, "val_acc": best_acc})
        print(f"epoch {epoch:3d}/{args.epochs}  loss {row['train_loss']:.4f}  train_acc {row['train_acc']:.4f}  "
              f"val_acc {val['acc']:.4f}  val_f1 {val['macro_f1']:.4f}  {row['time_s']:.0f}s"
              + ("  *" if is_best else ""), flush=True)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    save_checkpoint(run_dir / "last.pt", model, config, {"epoch": args.epochs})
    print(f"best val acc {best_acc:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
