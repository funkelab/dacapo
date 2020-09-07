import torch
from .loss import Loss


class CrossEntropyLoss(torch.nn.CrossEntropyLoss, Loss):

    def __init__(self, weight=None, ignore_label=-100):

        if weight is not None:
            weight = torch.tensor(weight)

        super(CrossEntropyLoss, self).__init__(weight, ignore_index=ignore_label)

    def forward(self, prediction, target):

        return super(CrossEntropyLoss, self).forward(prediction, target)
