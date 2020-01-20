import torch
import funlib.learn.torch as ft


class ShallowLearnerChain(torch.nn.Module):

    def __init__(
            self,
            fmaps_in,
            fmaps_out,
            num_latents,
            num_learners,
            create_shallow_learner):

        super(ShallowLearnerChain, self).__init__()

        # first learner from fmaps_in to latent
        learners = [create_shallow_learner(1, num_latents)]

        for i in range(num_learners - 1):
            learners.append(
                create_shallow_learner(
                    num_latents,
                    num_latents))
        self.learners = torch.nn.ModuleList(learners)

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(num_latents, 2, (1, 1)),
            torch.nn.Sigmoid()
        )

        self.num_learners = num_learners

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, x):
        x = self.learners[0](x)
        for i in range(1, self.num_learners):
            y = self.learners[i](x)
            x = self.crop(x, y.size()) + y
        return self.head(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
