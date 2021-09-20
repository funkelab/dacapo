from .helpers import Predictor

import gunpowder as gp

import torch
import numpy as np
import attr

from typing import Optional
from enum import Enum


# Define conv layers for different dimension counts
CONV_LAYERS = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


class AddClassLabels(gp.BatchFilter):
    def __init__(self, gt, target):
        self.gt = gt
        self.target = target

    def setup(self):
        self.provides(self.target, self.spec[self.gt].copy())
        self.enable_autoskip()

    def process(self, batch, request):
        spec = batch[self.gt].spec.copy()
        spec.dtype = np.int64
        batch[self.target] = gp.Array(batch[self.gt].data.astype(np.int64), spec)


class MaskToWeights(gp.BatchFilter):
    def __init__(self, mask, weights):
        self.mask = mask
        self.weights = weights

    def setup(self):
        self.provides(self.weights, self.spec[self.mask].copy())
        self.enable_autoskip()

    def process(self, batch, request):
        spec = batch[self.mask].spec.copy()
        spec.dtype = np.float32
        batch[self.weights] = gp.Array(batch[self.mask].data.astype(np.float32), spec)


@attr.s
class OneHotLabels(Predictor):

    # attributes that can be read from other configurable classes
    fmaps_in: Optional[int] = attr.ib(
        default=None,
        metadata={"help_text": "The number of feature maps provided by the model."},
    )
    dims: Optional[int] = attr.ib(
        default=None, metadata={"help_text": "The dimensionality of your data."}
    )  # can be read from data
    num_classes: Optional[int] = attr.ib(
        default=None,
        metadata={"help_text": "The number of classes in your ground truth data."},
    )  # can be read from data

    @property
    def target_fmaps(self):
        return 1

    @property
    def fmaps_out(self):
        return self.num_classes

    def head(self, architecture, dataset):
        return OneHotLabelsHead(
            dims=architecture.output_shape.dims,
            fmaps_in=architecture.fmaps_out,
            num_classes=dataset.gt.array_type.num_classes,
        )

    def add_target(self, gt, target, weights=None, mask=None):

        if mask is not None and weights is not None:
            weights_node = MaskToWeights(mask, weights)
        else:
            weights_node = None
        return AddClassLabels(gt, target), weights_node


class OneHotLabelsHead(torch.nn.Module):
    def __init__(self, dims: int, fmaps_in: int, num_classes: int):
        super(OneHotLabelsHead, self).__init__()

        conv = CONV_LAYERS[dims]
        logit_layers = [conv(fmaps_in, num_classes, (1,) * dims)]

        self.logits = torch.nn.Sequential(*logit_layers)
        self.probs = torch.nn.LogSoftmax()

    def forward(self, x):
        logits = self.logits(x)
        if not self.training:
            return self.probs(logits)
        return logits
