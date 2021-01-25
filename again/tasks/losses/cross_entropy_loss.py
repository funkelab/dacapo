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

        super(CrossEntropyLoss, self).__init__(
            weight,
            ignore_index=ignore_label,
            reduction='none')

    def forward(self, prediction, target, mask=None):

        have_neg_label = self.neg_label is not None
        have_mask = mask is not None

        if not have_neg_label and not have_mask:
            loss = super(CrossEntropyLoss, self).forward(prediction, target)
            return loss.mean()

        if have_neg_label:

            neg_mask = target == self.neg_label

            # replace neg_label with neg_target in target
            target[neg_mask] = self.neg_target

        # compute loss without reduction
        loss = super(CrossEntropyLoss, self).forward(prediction, target)

        # invert loss within neg_mask
        if have_neg_label:
            loss[neg_mask] *= -1

        # mask out
        if have_mask:
            loss[mask == 0] = 0

        return loss.mean()
