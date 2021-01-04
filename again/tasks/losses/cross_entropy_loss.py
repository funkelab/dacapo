import torch
from .loss import Loss


class CrossEntropyLoss(torch.nn.CrossEntropyLoss, Loss):

    def __init__(
            self,
            weight=None,
            ignore_label=-100,
            neg_label=None,
            neg_target=None):
        '''Args:

        neg_label, neg_target:

                Label ``neg_label`` is not ``neg_target``. Use this if you
                don't know the exact label, but you can clearly say it is not
                ``neg_target``.
        '''

        if weight is not None:
            weight = torch.tensor(weight)

        self.neg_label = neg_label
        self.neg_target = neg_target

        if self.neg_label is None:
            super(CrossEntropyLoss, self).__init__(weight, ignore_index=ignore_label)
        else:
            super(CrossEntropyLoss, self).__init__(
                weight,
                ignore_index=ignore_label,
                reduction='none')

    def forward(self, prediction, target):

        if self.neg_label is None:
            return super(CrossEntropyLoss, self).forward(prediction, target)

        # mask for all neg_labels
        neg_mask = target == self.neg_label

        # replace neg_label with neg_target in target
        target[neg_mask] = self.neg_target

        # print(target)
        # print(torch.unique(target))
        # print(prediction.shape)

        # compute loss without reduction
        loss = super(CrossEntropyLoss, self).forward(prediction, target)

        # invert loss within neg_mask
        loss[neg_mask] *= -1

        return loss.mean()
