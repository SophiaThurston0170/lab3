"""train.py with the additions for Exercises 2, 3 and 4d. train.py itself is not changed.

    python train_ext.py --model resnet18_small --img-size 48                    # Exercise 2
    python train_ext.py --epochs 80 --ema 0.999                                 # Exercise 3
    python train_ext.py --epochs 80 --ema 0.999 --mixup 0.2 --cutmix 1.0        # Exercise 3
    python train_ext.py --model facenet --img-size 160 --batch-size 64 --epochs 25 \
        --lr 1e-3 --backbone-lr-mult 0.1 --eval-batch-size 128                       # Exercise 4d

"""
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import evaluate_ext
import plots
from augment import augment, batches, prepare, to_device
from config_ext import get_config, run_name
from data import load_data
from models_ext import build_model, head_prefix
from utils import class_weights, get_device, loss_fn, predict, scores, set_seed


def mix_batch(x, y, cfg):
    """Mixup or CutMix on a prepared batch. y holds probability vectors, so the labels mix too."""
    if (cfg.mixup <= 0 and cfg.cutmix <= 0) or np.random.rand() >= cfg.mix_prob:
        return x, y
    use_cutmix = cfg.cutmix > 0 and (cfg.mixup <= 0 or np.random.rand() < 0.5)
    alpha = cfg.cutmix if use_cutmix else cfg.mixup
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if use_cutmix:
        # paste a rectangle from another face; its area sets how much of that face's label is used
        h, w = x.shape[-2:]
        cut = math.sqrt(1.0 - lam)
        ch, cw = int(h * cut), int(w * cut)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
        lam = 1.0 - (y1 - y0) * (x1 - x0) / (h * w)
    else:
        x = lam * x + (1.0 - lam) * x[perm]
    return x, lam * y + (1.0 - lam) * y[perm]


def main(argv=None):
    cfg = get_config(argv)
    set_seed(cfg.seed)
    device = get_device()

    data, class_names = load_data(cfg)          # may switch cfg.labels for the image-folder version
    cfg.run_dir = cfg.run_dir.parent / run_name(cfg)   # name may change if labels fell back to fer2013
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Run folder:", run_dir)
    num_classes = len(class_names)

    # ---- look at the data (saved as pictures) ----
    train_counts = data["train"][1].sum(0)
    plots.class_counts(train_counts, class_names, run_dir / "plots" / "class_counts.png")
    plots.sample_grid(data["train"][0], data["train"][1].argmax(1), class_names, run_dir / "plots" / "samples.png")
    gpu = to_device(data, device)
    x_train, y_train = gpu["train"]
    demo = augment(x_train[:16].float() / 255.0).cpu().numpy()[:, 0]
    plots.image_row(demo, "Augmented training images", run_dir / "plots" / "augmented.png")

    # ---- model, weighted loss, optimizer ----
    model = build_model(cfg.model, num_classes, cfg.pretrained).to(device)
    print(f"{cfg.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    weight = class_weights(train_counts, cfg.class_weighting, device)
    print(pd.DataFrame({"images": train_counts.round().astype(int), "loss weight": weight.cpu().numpy().round(2)},
                       index=class_names))

    prefix = head_prefix(cfg.model)
    body = [p for n, p in model.named_parameters() if not n.startswith(prefix)]
    head = [p for n, p in model.named_parameters() if n.startswith(prefix)]
    groups = [{"params": ps, "lr": lr} for ps, lr in [(body, cfg.lr * cfg.backbone_lr_mult), (head, cfg.lr)] if ps]
    
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = -(-len(x_train) // cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups], epochs=cfg.epochs,
                                                    steps_per_epoch=steps_per_epoch, pct_start=0.15)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # running average of the weights and BatchNorm statistics
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema), use_buffers=True) if cfg.ema > 0 else None
    eval_model = ema.module if ema is not None else model

    # ---- resume if a checkpoint exists ----
    start_epoch, best_score, history = 0, -1.0, []
    last_ckpt, best_ckpt = run_dir / "last.pt", run_dir / "best.pt"
    if last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        
        if ema is not None:
            ema.load_state_dict(ck["ema"])
            
        start_epoch, best_score, history = ck["epoch"] + 1, ck["best_score"], ck["history"]
        print(f"Resuming from epoch {start_epoch} (best validation {cfg.select_metric} so far: {best_score:.4f})")

    saved_cfg = {k: str(v) if k in ("data_dir", "run_dir") else v for k, v in vars(cfg).items()}
    with open(run_dir / "config.json", "w") as f:
        json.dump(saved_cfg, f, indent=2)

    # ---- training loop ----
    x_val, y_val = gpu["val"]
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0, total_loss, n_seen = time.time(), 0.0, 0
        for xb, yb in batches(x_train, y_train, cfg.batch_size, shuffle=True):
            xin, yb = mix_batch(prepare(xb, cfg.img_size, train=True, use_augment=cfg.augment), yb, cfg)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xin)
                loss = loss_fn(logits, yb, weight, cfg.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update_parameters(model)
            total_loss += loss.item() * len(xb)
            n_seen += len(xb)

        val_probs, val_labels, val_loss = predict(eval_model, x_val, y_val, cfg.img_size, cfg.tta,
                                                  batch_size=cfg.eval_batch_size,
                                                  weight=weight, label_smoothing=cfg.label_smoothing)
        val = scores(val_probs, val_labels)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / n_seen, "val_loss": val_loss, **val})
        improved = val[cfg.select_metric] > best_score
        if improved:
            best_score = val[cfg.select_metric]
            torch.save({"model": eval_model.state_dict(), "class_names": class_names, "config": saved_cfg,
                        "epoch": epoch + 1, "val": val}, best_ckpt)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None,
                    "epoch": epoch, "best_score": best_score, "history": history}, last_ckpt)
        print(f"epoch {epoch + 1:2d}/{cfg.epochs} | train loss {total_loss / n_seen:.3f} | val loss {val_loss:.3f} | "
              f"val acc {val['accuracy']:.3f} | val macro-F1 {val['macro_f1']:.3f} | {time.time() - t0:.0f}s"
              + ("  <- best" if improved else ""))

    hist = pd.DataFrame(history)
    if len(hist):
        hist.to_csv(run_dir / "history.csv", index=False)
        plots.learning_curves(hist, run_dir / "plots" / "learning_curves.png")

    # ---- final test-set evaluation with the best checkpoint ----
    evaluate_ext.main(["--checkpoint", str(best_ckpt), "--data-dir", str(cfg.data_dir)]
                      + ([] if cfg.tta else ["--no-tta"]), batch_size=cfg.eval_batch_size)


if __name__ == "__main__":
    main()
