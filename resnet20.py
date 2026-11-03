"""
resnet20.py
-----------
CIFAR-style ResNet-20 (He et al., "Deep Residual Learning for Image
Recognition", the 6n+2 family with n=3): a 3x3 stem conv, three stages of
three BasicBlocks each (16 / 32 / 64 channels, stride-2 at the first block
of stage 2 and 3), global average pool, one FC layer.

  1 (stem conv) + 3 stages * 3 blocks * 2 convs (=18) + 1 (fc) = 20 layers.

Every conv and the fc layer are QuantConv2d / QuantLinear -- drop-in
replacements for nn.Conv2d / nn.Linear that can be switched between:
  - full precision (bits=32, the default, i.e. plain float training)
  - uniform LSQ fake quantization at an arbitrary bitwidth
  - LVQ learnable-codebook vector quantization at an arbitrary bitwidth
      (num_levels = 2**bits)
per layer, at any time after construction -- which is exactly what the
mixed-precision pipeline needs (train float -> measure sensitivity -> pick
per-layer bitwidths -> switch each layer into its assigned quantization
scheme -> fine-tune).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_layers import QuantConv2d, QuantLinear, QuantAct, conv3x3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = QuantAct(bits=32)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act_out = QuantAct(bits=32)

        self.downsample = None
        self.pad_channels = 0
        if stride != 1 or in_planes != planes:
            # Option A (parameter-free) shortcut, as in the original
            # ResNet-20 CIFAR paper: stride-2 subsample + zero-pad extra
            # channels. Keeps the layer count at exactly 20 quantizable
            # layers (no extra 1x1 shortcut convs to account for).
            self.downsample = lambda x: F.avg_pool2d(x, 1, stride)
            self.pad_channels = planes - in_planes

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.act1(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)
            if self.pad_channels > 0:
                pad = self.pad_channels
                identity = F.pad(identity, (0, 0, 0, 0, pad // 2, pad - pad // 2))

        out = F.relu(out + identity)
        out = self.act_out(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.stem = conv3x3(3, 16)
        self.bn_stem = nn.BatchNorm2d(16)
        self.act_stem = QuantAct(bits=32)

        self.stage1 = self._make_stage(16, 3, stride=1)
        self.stage2 = self._make_stage(32, 3, stride=2)
        self.stage3 = self._make_stage(64, 3, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = QuantLinear(64, num_classes)

        self._init_weights()

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn_stem(self.stem(x)))
        out = self.act_stem(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out).flatten(1)
        return self.fc(out)

    def quantizable_layers(self):
        """Returns an ordered dict {name: module} of every QuantConv2d /
        QuantLinear in the network -- exactly the 20 layers whose
        sensitivity gets measured and whose bitwidth gets searched."""
        layers = {}
        for name, m in self.named_modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                layers[name] = m
        return layers

    def activation_pairs(self):
        """List of (conv_module, act_quantizer) pairs -- used by the
        pipeline to propagate each layer's searched weight bitwidth to
        the activation quantizer that follows it, without the pipeline
        needing to know this architecture's internal attribute names."""
        pairs = [(self.stem, self.act_stem)]
        for block in self.modules():
            if isinstance(block, BasicBlock):
                pairs.append((block.conv1, block.act1))
                pairs.append((block.conv2, block.act_out))
        return pairs


def resnet20(num_classes=10):
    return ResNet20(num_classes=num_classes)


if __name__ == "__main__":
    model = resnet20()
    layers = model.quantizable_layers()
    print(f"quantizable layers: {len(layers)}")
    for n in layers:
        print(" ", n)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("output shape:", y.shape)