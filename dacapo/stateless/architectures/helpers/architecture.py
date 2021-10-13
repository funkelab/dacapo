from funlib.geometry import Coordinate

import attr

from abc import ABC, abstractmethod


@attr.s
class Architecture(ABC):
    """
    The Architecture defines the backbone of the model that will be trained.
    The output of the Architecture will probably be some intermediate soft value
    that must be post processed to generate final results.

    Every architecture in DaCapo must define:
    - `module` method:
        A method that will produce a trainable torch module
    - `input_shape` property:
        The shape (excluding batch and channel dims) of the tensor that will
        be passed into your module
    - `output_shape` property:
        The shape (excluding batch and channel dims) of the tensor that will
        be returned from your module
    - `fmaps_out` property:
        The number of channels output from your module

    Additionally it is encouraged to provide support for varying input/output shapes
    during validation by overwriting the properties `eval_input_shape` and
    `eval_output_shape`. By default they will be the same as the shape used
    during training, however this can be extremely innefficient as you won't
    use the full gpu memory.

    Please note that abstract properties do not play well with attr.s classes
    so to ensure that your Architecture fits the desired API please add an example
    instance to the ARCHITECTURES fixture in
    "tests/fixtures/stateless/architectures/__init__.py"
    """
    name: str = attr.ib(
        metadata={"help_text": "Name of your model for easy search and reuse"}
    )

    @abstractmethod
    def module(self):
        pass

    @property
    def eval_input_shape(self) -> Coordinate:
        return self.input_shape

    @property
    def eval_output_shape(self) -> Coordinate:
        return self.output_shape