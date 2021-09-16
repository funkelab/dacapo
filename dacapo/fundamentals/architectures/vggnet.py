from .helpers import Architecture

from funlib.learn.torch.models import Vgg3D as VGGNetModule
import attr

from typing import List, Optional
from enum import Enum


@attr.s
class VGGNet(Architecture):
    # standard model attributes
    input_shape: List[int] = attr.ib(metadata={"help_text": "The input shape."})
    fmaps_out: int = attr.ib(
        metadata={"help_text": "The number of featuremaps provided."}
    )
    downsample_factors: List[List[int]] = attr.ib(
        metadata={
            "help_text": "The factor by which to downsample spatial dimensions along each axis."
        }
    )

    fmaps_in: Optional[int] = attr.ib(
        default=None, metadata={"help_text": "The number of channels in the input data"}
    )

    def module(self):
        return VGGNetModule(
            input_size=self.input_shape,
            fmaps=self.fmaps_in,
            output_classes=self.fmaps_out,
            downsample_factors=self.downsample_factors,
        )
