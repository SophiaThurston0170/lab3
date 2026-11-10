"""Extra networks for Exercises 2 and 4d. models.py itself is not changed.

- resnet18_small: ResNet-18 for 48x48 inputs (Exercise 2)
- facenet:        InceptionResNet-V1 pretrained for face recognition on VGGFace2 (Exercise 4d)
- farl:           FaRL ViT-B/16, pretrained on 20M face image-caption pairs (optional alternative)
Every other name is passed on to models.build_model unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from augment import IMAGENET_MEAN, IMAGENET_STD

# FaRL weights (Zheng et al., CVPR 2022), from https://github.com/FacePerceiver/FaRL
FARL_URL = ("https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/"
            "FaRL-Base-Patch16-LAIONFace20M-ep64.pth")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_model(name, num_classes, pretrained=True):
    if name == "resnet18_small":
        return resnet18_small(num_classes, pretrained)
    if name == "facenet":
        return FaceNetClassifier(num_classes, pretrained)
    if name == "farl":
        return FaRLClassifier(num_classes, pretrained)
    return models.build_model(name, num_classes, pretrained)


def head_prefix(name):
    return {"resnet18": "fc.", "resnet18_small": "fc.", "alexnet": "classifier.6.",
            "facenet": "net.logits.", "farl": "head.", "small_cnn": ""}[name]



def resnet18_small(num_classes, pretrained=True):
    
    model = models.build_model("resnet18", num_classes, pretrained)
    old = model.conv1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if pretrained:
        with torch.no_grad():
            model.conv1.weight.copy_(F.adaptive_avg_pool2d(old.weight, 3) * (49 / 9))
    model.maxpool = nn.Identity()
    return model



class FaceNetClassifier(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError as e:
            raise ImportError("The facenet model needs facenet-pytorch: pip install facenet-pytorch --no-deps "
                              f"(import failed with: {e})") from e
        self.net = InceptionResnetV1(pretrained="vggface2" if pretrained else None, classify=True,
                                     num_classes=num_classes, dropout_prob=0.6)
        for name, values in [("in_mean", IMAGENET_MEAN), ("in_std", IMAGENET_STD)]:
            self.register_buffer(name, torch.tensor(values).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        x01 = x * self.in_std + self.in_mean
        return self.net((x01 * 255.0 - 127.5) / 128.0)
