"""
CustomResNet — the kuih classifier model.

An ImageNet-pretrained ResNet-50 backbone with the early layers frozen and a
custom fully-connected classifier head. This is the single source of truth for
the architecture; the training and evaluation notebooks import it from here.

The default configuration matches the final competition model:
  - backbone: ResNet-50 (IMAGENET1K_V1 weights)
  - frozen: everything except `layer4` (set fine_tune_layer3=True to also
    fine-tune layer3, as in some experiment runs)
  - head:   2048 -> 1024 -> 512 -> 8, ReLU + Dropout between layers
"""

from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

# Canonical class order (alphabetical — matches sklearn LabelEncoder output)
CLASS_NAMES = [
    "Kek Lapis",
    "Kuih Kaswi Pandan",
    "Kuih Ketayap",
    "Kuih Lapis",
    "Kuih Seri Muka",
    "Kuih Talam",
    "Kuih Ubi Kayu",
    "Onde-Onde",
]


class CustomResNet(nn.Module):
    def __init__(self, num_classes: int = 8, dropout: float = 0.4455,
                 fine_tune_layer3: bool = False):
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze the backbone, then selectively unfreeze the deepest stage(s).
        trainable = ("layer4",) if not fine_tune_layer3 else ("layer3", "layer4")
        for name, param in self.resnet.named_parameters():
            param.requires_grad = any(t in name for t in trainable)

        # Replace the original 1000-way FC with an identity so we get the raw
        # 2048-dim feature vector, then attach our own head.
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.resnet(x))
