from .helpers import Architecture

from funlib.learn.torch.models import Vgg3D as VGGNetModule
from funlib.geometry import Coordinate

import attr
import torch

from typing import List, Optional
from enum import Enum


@attr.s
class VGGNet(Architecture):
    # standard model attributes
    input_shape: Coordinate = attr.ib(metadata={"help_text": "The input shape."})
    fmaps_out: int = attr.ib(
        metadata={"help_text": "The number of featuremaps provided."}
    )
    fmap_inc: List[int] = attr.ib(
        metadata={
            "help_text": "The factors by which to increment channels at each downsample"
        }
    )
    n_convolutions: List[int] = attr.ib()
    downsample_factors: List[Coordinate] = attr.ib(
        metadata={
            "help_text": "The factor by which to downsample spatial dimensions along each axis."
        }
    )
    num_dense: Optional[int] = attr.ib()
    fmaps_in: Optional[int] = attr.ib(
        default=None, metadata={"help_text": "The number of channels in the input data"}
    )

    @property
    def output_shape(self):
        """
        VGGNet outputs a classification for each passed in Roi.
        The output can be thought of as having shape 1 for each
        axis.
        """
        return Coordinate((1,) * self.input_shape.dims)

    def module(self):
        return torch.nn.Sequential(
            VGGNetModule(
                input_size=self.input_shape,
                fmap_inc=self.fmap_inc,
                n_convolutions=self.n_convolutions,
                fmaps=self.fmaps_in,
                output_classes=self.fmaps_out,
                downsample_factors=self.downsample_factors,
                input_fmaps=self.fmaps_in,
                num_dense=self.num_dense,
            ),
            AddSpatial(*self.output_shape),
        )


class AddSpatial(torch.nn.Module):
    def __init__(self, *args):
        super(AddSpatial, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape + self.shape)
