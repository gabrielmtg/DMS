"""Modelo do classificador de celular (PyTorch). Só treino e exportação importam este módulo."""

import torch
from torch import nn
from torchvision.io import ImageReadMode, decode_image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PhoneNet(nn.Module):
    """MobileNetV3-Small com a normalização dentro: entrada NCHW float de 0 a 255."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = mobilenet_v3_small(weights=weights)
        last = self.net.classifier[-1]
        self.net.classifier[-1] = nn.Linear(last.in_features, num_classes)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)

    def set_trainable(self, backbone: bool) -> None:
        """A cabeça sempre treina; o backbone só no fine-tune, e o BatchNorm nunca."""
        for param in self.net.features.parameters():
            param.requires_grad = backbone
        for module in self.net.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Com poucos motoristas, recalcular as médias do BatchNorm as deixaria
        # instáveis; ficam as do ImageNet.
        for module in self.net.features.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self


class ExportWrapper(nn.Module):
    """O que vai para o .tflite: NHWC de 0 a 255 -> probabilidades."""

    def __init__(self, model: PhoneNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x.permute(0, 3, 1, 2)), dim=1)


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = [str(path) for path in paths]
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        image = decode_image(self.paths[i], mode=ImageReadMode.RGB)
        return self.transform(image), self.labels[i]


def train_transform():
    return v2.Compose(
        [
            # Vale porque Left e Right foram unidos em uma classe só.
            v2.RandomHorizontalFlip(),
            # Pequena variação de escala e posição, como mexer no scale e no shift do crop.
            v2.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToDtype(torch.float32),
        ]
    )


def eval_transform():
    return v2.ToDtype(torch.float32)
