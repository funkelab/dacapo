from .helpers import Loss

import torch
import attr


@attr.s
class MSELoss(Loss):

    def module(self):
        return torch.nn.MSELoss()
