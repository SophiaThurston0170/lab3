"""Settings for train_ext.py: everything in config.py, plus options for Exercises 2, 3 and 4d.

config.py itself is not changed. New options:

    --model resnet18_small   ResNet-18 adapted to 48x48 faces (Exercise 2; use with --img-size 48)
    --model facenet          InceptionResNet-V1 pretrained for face recognition (Exercise 4d; use with --img-size 160)
    --model farl             FaRL ViT-B/16 pretrained on 20M face images (optional; use with --img-size 224)
    --ema 0.999              keep a running average of the weights and evaluate that (Exercise 3)
    --mixup 0.2              mixup (Exercise 3)
    --cutmix 1.0             CutMix (Exercise 3)
    --backbone-lr-mult 0.05  smaller learning rate for pretrained layers than for the new classifier
    --tag text               extra text in the run folder name

Run folder names also include the number of epochs, "scratch" for --no-pretrained, and the
options above.
"""
import argparse
from pathlib import Path

import config

DEFAULTS = dict(config.DEFAULTS)
DEFAULTS.update(
    ema=0.0,                 # weight-EMA decay, e.g. 0.999; 0 = off
    mixup=0.0,               # mixup Beta(alpha, alpha), e.g. 0.2; 0 = off
    cutmix=0.0,              # CutMix Beta(alpha, alpha), e.g. 1.0; 0 = off
    mix_prob=0.5,            # share of training batches that get mixup / CutMix (if enabled)
    backbone_lr_mult=1.0,    # pretrained layers train at lr x this; the new classifier at lr
    eval_batch_size=512,     # lower it (e.g. 128) for big models if the GPU runs out of memory
    tag="",                  # extra text for the run folder name
)

HELP = dict(config.HELP)
HELP.update(
    ema="EMA decay for a running average of the weights (e.g. 0.999); 0 turns it off",
    mixup="mixup alpha (e.g. 0.2); 0 turns it off",
    cutmix="CutMix alpha (e.g. 1.0); 0 turns it off",
    mix_prob="fraction of training batches that are mixed when mixup/CutMix is on",
    backbone_lr_mult="learning rate multiplier for the pretrained layers (the new classifier uses --lr)",
    eval_batch_size="batch size for validation/test predictions",
    tag="extra text added to the run folder name",
)

CHOICES = {k: list(v) for k, v in config.CHOICES.items()}
CHOICES["model"] = CHOICES["model"] + ["resnet18_small", "facenet", "farl"]


def get_config(argv=None):
    parser = argparse.ArgumentParser(description="Extended FER2013 training (Exercises 2, 3 and 4d)")
    for name, default in DEFAULTS.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=HELP.get(name))
        else:
            parser.add_argument(flag, type=type(default), default=default, choices=CHOICES.get(name),
                                help=f"{HELP.get(name, '')} (default: {default})")
    cfg = parser.parse_args(argv)
    cfg.data_dir = Path(cfg.data_dir)
    cfg.run_dir = Path(cfg.out_dir) / run_name(cfg)
    return cfg


def run_name(cfg):
    parts = [config.run_name(cfg)]
    if not cfg.pretrained and cfg.model != "small_cnn":
        parts.append("scratch")
    parts.append(f"e{cfg.epochs}")
    if cfg.ema > 0:
        parts.append(f"ema{cfg.ema:g}")
    if cfg.mixup > 0:
        parts.append(f"mixup{cfg.mixup:g}")
    if cfg.cutmix > 0:
        parts.append(f"cutmix{cfg.cutmix:g}")
    if cfg.backbone_lr_mult != 1.0:
        parts.append(f"blr{cfg.backbone_lr_mult:g}")
    if cfg.tag:
        parts.append(cfg.tag)
    return "_".join(parts)
